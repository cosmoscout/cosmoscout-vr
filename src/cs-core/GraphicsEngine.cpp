////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "GraphicsEngine.hpp"

#include "../cs-graphics/ClearHDRBufferNode.hpp"
#include "../cs-graphics/EclipseShadowMap.hpp"
#include "../cs-graphics/SetupGLNode.hpp"
#include "../cs-graphics/TextureLoader.hpp"
#include "../cs-graphics/ToneMappingNode.hpp"
#include "../cs-utils/utils.hpp"
#include "logger.hpp"

#include <GL/glew.h>
#include <VistaKernel/DisplayManager/VistaDisplayManager.h>
#include <VistaKernel/DisplayManager/VistaProjection.h>
#include <VistaKernel/DisplayManager/VistaViewport.h>
#include <VistaKernel/DisplayManager/VistaWindow.h>
#include <VistaKernel/GraphicsManager/VistaOpenGLNode.h>
#include <VistaKernel/GraphicsManager/VistaSceneGraph.h>
#include <VistaKernel/VistaSystem.h>
#include <VistaKernelOpenSGExt/VistaOpenSGMaterialTools.h>

namespace cs::core {

////////////////////////////////////////////////////////////////////////////////////////////////////

void GLAPIENTRY MessageCallback(GLenum source, GLenum type, GLuint id, GLenum severity,
    GLsizei length, const GLchar* message, const void* userParams) {

  // get the log level from the settings
  const cs::core::Settings* settings = reinterpret_cast<const cs::core::Settings*>(userParams);

  // Print the following infos (OpenGL errors, shader compile errors, perf. warnings, shader
  // compilation warnings, depricated code, redundant state changes, undefined behaviour, anything
  // that isnt an error or perf. issue)
  if (settings->pLogLevelGL.get() <= spdlog::level::debug &&
      severity == GL_DEBUG_SEVERITY_NOTIFICATION) {
    logger().debug("{} (source=0x{:x} type=0x{:x} id=0x{:x})", message, source, type, id);
    return;
  }

  // Print the following infos (OpenGL errors, shader compile errors, perf. warnings, shader
  // compilation warnings, depricated code, redundant state changes, undefined behaviour)
  if (settings->pLogLevelGL.get() <= spdlog::level::info && severity == GL_DEBUG_SEVERITY_LOW) {
    logger().info("{} (source=0x{:x} type=0x{:x} id=0x{:x})", message, source, type, id);
    return;
  }

  // Print the following infos (OpenGL errors, shader compile errors, perf. warnings, shader
  // compilation warnings, depricated code)
  if (settings->pLogLevelGL.get() <= spdlog::level::warn && severity == GL_DEBUG_SEVERITY_MEDIUM) {
    logger().warn("{} (source=0x{:x} type=0x{:x} id=0x{:x})", message, source, type, id);
    return;
  }

  // Print the following infos (OpenGL errors, shader compile errors)
  if (settings->pLogLevelGL.get() <= spdlog::level::critical &&
      severity == GL_DEBUG_SEVERITY_HIGH) {
    logger().error("{} (source=0x{:x} type=0x{:x} id=0x{:x})", message, source, type, id);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

GraphicsEngine::GraphicsEngine(std::shared_ptr<core::Settings> settings)
    : mSettings(std::move(settings))
    , mShadowMap(std::make_shared<graphics::ShadowMap>())
    , mFallbackEclipseShadowMap(
          graphics::TextureLoader::loadFromFile("../share/resources/textures/fallbackShadow.tif")) {

  // Tell the user what's going on.
  logger().debug("Creating GraphicsEngine.");
  logger().info("OpenGL Vendor:  {}", reinterpret_cast<const char*>(glGetString(GL_VENDOR)));
  logger().info("OpenGL Version: {}", reinterpret_cast<const char*>(glGetString(GL_VERSION)));

  auto* pSG = GetVistaSystem()->GetGraphicsManager()->GetSceneGraph();

  mSettings->mGraphics.pEnableVsync.connect([](bool value) {
    GetVistaSystem()
        ->GetDisplayManager()
        ->GetWindows()
        .begin()
        ->second->GetWindowProperties()
        ->SetVSyncEnabled(value);
  });

  // setup OpenGL debugging ------------------------------------------------------------------------

  mSettings->pLogLevelGL.connectAndTouch([](auto level) {
    if (level == spdlog::level::off) {
      glDisable(GL_DEBUG_OUTPUT);
    } else {
      glEnable(GL_DEBUG_OUTPUT);
    }
  });

  // Attach the debug callback to print the messages.
  glDebugMessageCallback(MessageCallback, static_cast<void*>(mSettings.get()));

  // Ignore debug messages telling us buffers are moved in memory.
  GLuint id = 131186;
  glDebugMessageControl(0x8246, 0x8250, GL_DONT_CARE, 1, &id, GL_FALSE);

  // Ignore synchronized transfer warning.
  id = 0x20052;
  glDebugMessageControl(0x8246, 0x8250, GL_DONT_CARE, 1, &id, GL_FALSE);

  // setup shadows ---------------------------------------------------------------------------------

  mShadowMap->setEnabled(false);
  mShadowMap->setResolution(static_cast<uint32_t>(mSettings->mGraphics.pShadowMapResolution.get()));
  mShadowMap->setBias(mSettings->mGraphics.pShadowMapBias.get() * 0.0001F);
  pSG->NewOpenGLNode(pSG->GetRoot(), mShadowMap.get());

  calculateCascades();

  mSettings->mGraphics.pEnableShadows.connect([this](bool val) { mShadowMap->setEnabled(val); });

  mSettings->mGraphics.pEnableShadowsFreeze.connect(
      [this](bool val) { mShadowMap->setFreezeCascades(val); });

  mSettings->mGraphics.pShadowMapResolution.connect(
      [this](int val) { mShadowMap->setResolution(static_cast<uint32_t>(val)); });

  mSettings->mGraphics.pShadowMapCascades.connect([this](int /*unused*/) { calculateCascades(); });

  mSettings->mGraphics.pShadowMapBias.connect(
      [this](float val) { mShadowMap->setBias(val * 0.0001f); });

  mSettings->mGraphics.pShadowMapSplitDistribution.connect(
      [this](float /*unused*/) { calculateCascades(); });

  mSettings->mGraphics.pShadowMapRange.connect(
      [this](glm::vec2 /*unused*/) { calculateCascades(); });

  mSettings->mGraphics.pShadowMapExtension.connect(
      [this](glm::vec2 /*unused*/) { calculateCascades(); });

  // setup eclipse shadows -------------------------------------------------------------------------

  // Load the eclipse shadow maps of all configured bodies. If a body has no specific texture, the
  // fallback texture is used instead.
  if (mSettings->mGraphics.mEclipseShadowMaps.has_value()) {
    for (auto const& s : mSettings->mGraphics.mEclipseShadowMaps.value()) {
      auto shadowMap       = std::make_shared<graphics::EclipseShadowMap>();
      shadowMap->mOccluder = s.first;

      if (s.second.mTexture) {
        shadowMap->mTexture = graphics::TextureLoader::loadFromFile(*s.second.mTexture);
        shadowMap->mTexture->SetWrapS(GL_CLAMP_TO_EDGE);
        shadowMap->mTexture->SetWrapT(GL_CLAMP_TO_EDGE);
      } else {
        shadowMap->mTexture = mFallbackEclipseShadowMap;
      }

      mEclipseShadowMaps.push_back(shadowMap);
    }
  }

  // Also, the fallback shadow map should use texture clamping.
  mFallbackEclipseShadowMap->SetWrapS(GL_CLAMP_TO_EDGE);
  mFallbackEclipseShadowMap->SetWrapT(GL_CLAMP_TO_EDGE);

  // setup HDR buffer ------------------------------------------------------------------------------
  int multiSamples = GetVistaSystem()
                         ->GetDisplayManager()
                         ->GetWindows()
                         .begin()
                         ->second->GetWindowProperties()
                         ->GetMultiSamples();

  mHDRBuffer = std::make_shared<graphics::HDRBuffer>(multiSamples);

  // Create a node which clears the HDRBuffer at the beginning of a frame (this will be enabled only
  // if HDR rendering is enabled).
  mClearNode        = std::make_shared<graphics::ClearHDRBufferNode>(mHDRBuffer);
  auto* clearGLNode = pSG->NewOpenGLNode(pSG->GetRoot(), mClearNode.get());
  VistaOpenSGMaterialTools::SetSortKeyOnSubtree(
      clearGLNode, static_cast<int>(utils::DrawOrder::eClearHDRBuffer));

  mSetupGLNode      = std::make_shared<graphics::SetupGLNode>();
  auto* setupGLNode = pSG->NewOpenGLNode(pSG->GetRoot(), mSetupGLNode.get());
  VistaOpenSGMaterialTools::SetSortKeyOnSubtree(
      setupGLNode, static_cast<int>(utils::DrawOrder::eSetupOpenGL));

  // Create a node which performas tonemapping of the HDRBuffer at the end of a frame (this will be
  // enabled only if HDR rendering is enabled).
  mToneMappingNode        = std::make_shared<graphics::ToneMappingNode>(mHDRBuffer);
  auto* toneMappingGLNode = pSG->NewOpenGLNode(pSG->GetRoot(), mToneMappingNode.get());
  VistaOpenSGMaterialTools::SetSortKeyOnSubtree(
      toneMappingGLNode, static_cast<int>(utils::DrawOrder::eToneMapping));

  mSettings->mGraphics.pGlareIntensity.connectAndTouch(
      [this](float val) { mToneMappingNode->setGlareIntensity(val); });

  mSettings->mGraphics.pGlareQuality.connectAndTouch(
      [this](uint32_t val) { mHDRBuffer->setGlareQuality(val); });

  mSettings->mGraphics.pGlareMode.connectAndTouch(
      [this](graphics::HDRBuffer::GlareMode mode) { mHDRBuffer->setGlareMode(mode); });

  mSettings->mGraphics.pToneMappingMode.connectAndTouch(
      [this](graphics::ToneMappingNode::ToneMappingMode mode) {
        mToneMappingNode->setToneMappingMode(mode);
      });

  mSettings->mGraphics.pEnableBicubicGlareFilter.connectAndTouch(
      [this](bool enable) { mHDRBuffer->setEnableBicubicGlareFilter(enable); });

  mSettings->mGraphics.pEnable32BitGlare.connectAndTouch(
      [this](bool enable) { mHDRBuffer->setEnable32BitGlare(enable); });

  mSettings->mGraphics.pExposureCompensation.connectAndTouch(
      [this](float val) { mToneMappingNode->setExposureCompensation(val); });

  mSettings->mGraphics.pExposureAdaptionSpeed.connectAndTouch(
      [this](float val) { mToneMappingNode->setExposureAdaptionSpeed(val); });

  mSettings->mGraphics.pAutoExposureRange.connectAndTouch([this](glm::vec2 val) {
    mToneMappingNode->setMinAutoExposure(val[0]);
    mToneMappingNode->setMaxAutoExposure(val[1]);
  });

  mSettings->mGraphics.pEnableHDR.connectAndTouch([clearGLNode, toneMappingGLNode](bool enabled) {
    clearGLNode->SetIsEnabled(enabled);
    toneMappingGLNode->SetIsEnabled(enabled);
  });

  mSettings->mGraphics.pEnableAutoExposure.connectAndTouch(
      [this](bool enabled) { mToneMappingNode->setEnableAutoExposure(enabled); });

  mSettings->mGraphics.pExposure.connectAndTouch([this](float value) {
    if (!mSettings->mGraphics.pEnableAutoExposure.get()) {
      mToneMappingNode->setExposure(value);
    }
  });
}

////////////////////////////////////////////////////////////////////////////////////////////////////

GraphicsEngine::~GraphicsEngine() {
  try {
    // Tell the user what's going on.
    logger().debug("Deleting GraphicsEngine.");
  } catch (...) {}
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void GraphicsEngine::registerCaster(graphics::ShadowCaster* caster) {
  mShadowMap->registerCaster(caster);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void GraphicsEngine::unregisterCaster(graphics::ShadowCaster* caster) {
  mShadowMap->deregisterCaster(caster);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void GraphicsEngine::update(glm::vec3 const& sunDirection) {
  mShadowMap->setSunDirection(VistaVector3D(sunDirection.x, sunDirection.y, sunDirection.z));

  // Update projection. When the sensor size control is enabled, we will calculate the projection
  // plane extents based on the screens aspect ratio, the given sensor diagonal and sensor focal
  // length.
  if (mSettings->pEnableSensorSizeControl.get()) {
    VistaViewport* pViewport(GetVistaSystem()->GetDisplayManager()->GetViewports().begin()->second);
    int            sizeX = 0;
    int            sizeY = 0;
    pViewport->GetViewportProperties()->GetSize(sizeX, sizeY);
    double aspect = 1.0 * sizeX / sizeY;

    VistaProjection::VistaProjectionProperties* pProjProps =
        pViewport->GetProjection()->GetProjectionProperties();

    double height =
        mSettings->mGraphics.pSensorDiagonal.get() / std::sqrt(std::pow(aspect, 2.0) + 1.0);
    height /= mSettings->mGraphics.pFocalLength.get();
    double width = aspect * height;
    pProjProps->SetProjPlaneExtents(-width / 2, width / 2, -height / 2, height / 2);
    pProjProps->SetProjPlaneMidpoint(0, 0, -1);
  }

  // Update exposure. If auto exposure is enabled, the property will reflect the exposure chosen by
  // the tonemapping node.
  if (mSettings->mGraphics.pEnableAutoExposure.get()) {
    mSettings->mGraphics.pExposure = mToneMappingNode->getExposure();
  }

  if (mSettings->mGraphics.pEnableHDR.get()) {
    pAverageLuminance = mToneMappingNode->getLastAverageLuminance();
    pMaximumLuminance = mToneMappingNode->getLastMaximumLuminance();
  }

  for (auto& viewport : mDepthBuffers) {
    viewport.second.mDirty = true;
  }

  for (auto& viewport : mColorBuffers) {
    viewport.second.mDirty = true;
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::shared_ptr<graphics::ShadowMap> GraphicsEngine::getShadowMap() const {
  return mShadowMap;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::shared_ptr<graphics::HDRBuffer> GraphicsEngine::getHDRBuffer() const {
  return mHDRBuffer;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::vector<std::shared_ptr<graphics::EclipseShadowMap>> const&
GraphicsEngine::getEclipseShadowMaps() const {
  return mEclipseShadowMaps;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

VistaTexture* GraphicsEngine::getCurrentDepthBufferAsTexture(bool forceCopy) {
  if (mSettings->mGraphics.pEnableHDR.get()) {
    return mHDRBuffer->getDepthAttachment();
  }

  auto* viewport = GetVistaSystem()->GetDisplayManager()->GetCurrentRenderInfo()->m_pViewport;
  auto  it       = mDepthBuffers.find(viewport);

  if (it == mDepthBuffers.end()) {
    ViewportData data;
    data.mBuffer = std::make_shared<VistaTexture>(GL_TEXTURE_2D);
    data.mBuffer->SetWrapS(GL_CLAMP);
    data.mBuffer->SetWrapT(GL_CLAMP);
    data.mBuffer->SetMinFilter(GL_NEAREST);
    data.mBuffer->SetMagFilter(GL_NEAREST);

    mDepthBuffers[viewport] = std::move(data);
    it                      = mDepthBuffers.find(viewport);
  }

  if (it->second.mDirty || forceCopy) {
    int x, y, w, h;
    viewport->GetViewportProperties()->GetPosition(x, y);
    viewport->GetViewportProperties()->GetSize(w, h);
    it->second.mBuffer->Bind();
    glCopyTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, x, y, w, h, 0);
    it->second.mBuffer->Unbind();
    it->second.mDirty = false;
  }

  return it->second.mBuffer.get();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

VistaTexture* GraphicsEngine::getCurrentColorBufferAsTexture(bool forceCopy) {
  if (mSettings->mGraphics.pEnableHDR.get()) {
    return mHDRBuffer->getCurrentReadAttachment();
  }

  auto* viewport = GetVistaSystem()->GetDisplayManager()->GetCurrentRenderInfo()->m_pViewport;
  auto  it       = mColorBuffers.find(viewport);

  if (it == mColorBuffers.end()) {
    ViewportData data;
    data.mBuffer = std::make_shared<VistaTexture>(GL_TEXTURE_2D);
    data.mBuffer->SetWrapS(GL_CLAMP);
    data.mBuffer->SetWrapT(GL_CLAMP);
    data.mBuffer->SetMinFilter(GL_NEAREST);
    data.mBuffer->SetMagFilter(GL_NEAREST);

    mColorBuffers[viewport] = std::move(data);
    it                      = mColorBuffers.find(viewport);
  }

  if (it->second.mDirty || forceCopy) {
    int x, y, w, h;
    viewport->GetViewportProperties()->GetPosition(x, y);
    viewport->GetViewportProperties()->GetSize(w, h);
    it->second.mBuffer->Bind();
    glCopyTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, x, y, w, h, 0);
    it->second.mBuffer->Unbind();
    it->second.mDirty = false;
  }

  return it->second.mBuffer.get();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void GraphicsEngine::calculateCascades() {
  float              nearEnd = mSettings->mGraphics.pShadowMapRange.get().x;
  float              farEnd  = mSettings->mGraphics.pShadowMapRange.get().y;
  int                count   = mSettings->mGraphics.pShadowMapCascades.get();
  std::vector<float> splits(count + 1);
  for (size_t i(0); i < splits.size(); ++i) {
    float alpha = static_cast<float>(i) / static_cast<float>(count);
    alpha       = std::pow(alpha, mSettings->mGraphics.pShadowMapSplitDistribution.get());
    splits[i]   = glm::mix(nearEnd, farEnd, alpha);
  }
  mShadowMap->setCascadeSplits(splits);
  mShadowMap->setSunNearClipOffset(mSettings->mGraphics.pShadowMapExtension.get().x);
  mShadowMap->setSunFarClipOffset(mSettings->mGraphics.pShadowMapExtension.get().y);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cs::core
