////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "GraphicsEngine.hpp"

#include "../cs-graphics/ClearHDRBufferNode.hpp"
#include "../cs-graphics/HDRBuffer.hpp"
#include "../cs-graphics/ToneMappingNode.hpp"
#include "../cs-utils/utils.hpp"

#include <GL/glew.h>
#include <VistaKernel/DisplayManager/VistaDisplayManager.h>
#include <VistaKernel/DisplayManager/VistaProjection.h>
#include <VistaKernel/DisplayManager/VistaViewport.h>
#include <VistaKernel/GraphicsManager/VistaOpenGLNode.h>
#include <VistaKernel/GraphicsManager/VistaSceneGraph.h>
#include <VistaKernel/VistaSystem.h>
#include <VistaKernelOpenSGExt/VistaOpenSGMaterialTools.h>
#include <spdlog/spdlog.h>

namespace cs::core {

////////////////////////////////////////////////////////////////////////////////////////////////////

GraphicsEngine::GraphicsEngine(std::shared_ptr<core::Settings> const& settings)
    : mSettings(settings)
    , mShadowMap(std::make_shared<graphics::ShadowMap>()) {

  // Tell the user what's going on.
  spdlog::debug("Creating GraphicsEngine.");
  spdlog::info("OpenGL Vendor:  {}", glGetString(GL_VENDOR));
  spdlog::info("OpenGL Version: {}", glGetString(GL_VERSION));

  auto pSG = GetVistaSystem()->GetGraphicsManager()->GetSceneGraph();

  // setup shadows ---------------------------------------------------------------------------------

  mShadowMap->setEnabled(false);
  mShadowMap->setResolution((uint32_t)mSettings->mGraphics.pShadowMapResolution.get());
  mShadowMap->setBias(mSettings->mGraphics.pShadowMapBias.get() * 0.0001f);
  pSG->NewOpenGLNode(pSG->GetRoot(), mShadowMap.get());

  calculateCascades();

  mSettings->mGraphics.pEnableShadows.connect([this](bool val) { mShadowMap->setEnabled(val); });

  mSettings->mGraphics.pEnableShadowsFreeze.connect(
      [this](bool val) { mShadowMap->setFreezeCascades(val); });

  mSettings->mGraphics.pShadowMapResolution.connect(
      [this](int val) { mShadowMap->setResolution((uint32_t)val); });

  mSettings->mGraphics.pShadowMapCascades.connect([this](int) { calculateCascades(); });

  mSettings->mGraphics.pShadowMapBias.connect(
      [this](float val) { mShadowMap->setBias(val * 0.0001f); });

  mSettings->mGraphics.pShadowMapSplitDistribution.connect([this](float) { calculateCascades(); });

  mSettings->mGraphics.pShadowMapRange.connect([this](glm::vec2) { calculateCascades(); });

  mSettings->mGraphics.pShadowMapExtension.connect([this](glm::vec2) { calculateCascades(); });

  // setup HDR buffer ------------------------------------------------------------------------------

  mHDRBuffer = std::make_shared<graphics::HDRBuffer>();

  // Create a node which clears the HDRBuffer at the beginning of a frame (this will be enabled only
  // if HDR rendering is enabled).
  mClearNode       = std::make_shared<graphics::ClearHDRBufferNode>(mHDRBuffer);
  auto clearGLNode = pSG->NewOpenGLNode(pSG->GetRoot(), mClearNode.get());
  VistaOpenSGMaterialTools::SetSortKeyOnSubtree(
      clearGLNode, static_cast<int>(utils::DrawOrder::eClearHDRBuffer));

  // Create a node which performas tonemapping of the HDRBuffer at the end of a frame (this will be
  // enabled only if HDR rendering is enabled).
  mToneMappingNode       = std::make_shared<graphics::ToneMappingNode>(mHDRBuffer);
  auto toneMappingGLNode = pSG->NewOpenGLNode(pSG->GetRoot(), mToneMappingNode.get());
  VistaOpenSGMaterialTools::SetSortKeyOnSubtree(
      toneMappingGLNode, static_cast<int>(utils::DrawOrder::eToneMapping));

  mSettings->mGraphics.pGlowIntensity.connectAndTouch(
      [this](float val) { mToneMappingNode->setGlowIntensity(val); });

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

    // Whenever the exposure changes, and if auto-glow is enabled, we change the glow intensity
    // based on the exposure value. The auto-glow amount is based on the current exposure relative
    // to the auto-exposure range.
    if (mSettings->mGraphics.pEnableAutoGlow.get()) {
      float glow = (mSettings->mGraphics.pAutoExposureRange.get()[0] - value) /
                   (mSettings->mGraphics.pAutoExposureRange.get()[0] -
                       mSettings->mGraphics.pAutoExposureRange.get()[1]);
      mSettings->mGraphics.pGlowIntensity = std::clamp(glow * 0.5f, 0.001f, 1.f);
    }
  });
}

////////////////////////////////////////////////////////////////////////////////////////////////////

GraphicsEngine::~GraphicsEngine() {
  // Tell the user what's going on.
  spdlog::debug("Deleting GraphicsEngine.");
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
    float aspect = 1.f * sizeX / sizeY;

    VistaProjection::VistaProjectionProperties* pProjProps =
        pViewport->GetProjection()->GetProjectionProperties();

    float height =
        mSettings->mGraphics.pSensorDiagonal.get() / std::sqrt(std::pow(aspect, 2.f) + 1.f);
    height /= mSettings->mGraphics.pFocalLength.get();
    float width = aspect * height;
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

bool glDebugOnlyErrors = true;

void GLAPIENTRY oglMessageCallback(GLenum source, GLenum type, GLuint id, GLenum severity,
    GLsizei length, const GLchar* message, const void* userParam) {
  if (type == GL_DEBUG_TYPE_ERROR)
    fprintf(
        stderr, "GL ERROR: type = 0x%x, severity = 0x%x, message = %s\n", type, severity, message);
  else if (!glDebugOnlyErrors)
    fprintf(stdout, "GL WARNING: type = 0x%x, severity = 0x%x, message = %s\n", type, severity,
        message);
}

void GraphicsEngine::enableGLDebug(bool onlyErrors) {
  glDebugOnlyErrors = onlyErrors;
  glEnable(GL_DEBUG_OUTPUT);
  glDebugMessageCallback(oglMessageCallback, nullptr);
}

void GraphicsEngine::disableGLDebug() {
  glDisable(GL_DEBUG_OUTPUT);
  glDebugMessageCallback(nullptr, nullptr);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void GraphicsEngine::calculateCascades() {
  float              nearEnd = mSettings->mGraphics.pShadowMapRange.get().x;
  float              farEnd  = mSettings->mGraphics.pShadowMapRange.get().y;
  int                count   = mSettings->mGraphics.pShadowMapCascades.get();
  std::vector<float> splits(count + 1);
  for (int i(0); i < splits.size(); ++i) {
    float alpha = (float)(i) / count;
    alpha       = std::pow(alpha, mSettings->mGraphics.pShadowMapSplitDistribution.get());
    splits[i]   = glm::mix(nearEnd, farEnd, alpha);
  }
  mShadowMap->setCascadeSplits(splits);
  mShadowMap->setSunNearClipOffset(mSettings->mGraphics.pShadowMapExtension.get().x);
  mShadowMap->setSunFarClipOffset(mSettings->mGraphics.pShadowMapExtension.get().y);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cs::core
