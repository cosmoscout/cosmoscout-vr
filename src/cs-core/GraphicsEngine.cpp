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

GraphicsEngine::GraphicsEngine(std::shared_ptr<const core::Settings> const& settings)
    : mSettings(settings)
    , mShadowMap(std::make_shared<graphics::ShadowMap>()) {

  // Tell the user what's going on.
  spdlog::debug("Creating GraphicsEngine.");
  spdlog::info("OpenGL Vendor:  {}", glGetString(GL_VENDOR));
  spdlog::info("OpenGL Version: {}", glGetString(GL_VERSION));

  auto pSG = GetVistaSystem()->GetGraphicsManager()->GetSceneGraph();

  pWidgetScale = settings->mWidgetScale;
  pEnableHDR   = settings->mEnableHDR.value_or(false);

  // setup shadows ---------------------------------------------------------------------------------

  mShadowMap->setEnabled(false);
  mShadowMap->setResolution((uint32_t)pShadowMapResolution.get());
  mShadowMap->setBias(pShadowMapBias.get() * 0.0001f);
  pSG->NewOpenGLNode(pSG->GetRoot(), mShadowMap.get());

  calculateCascades();

  pEnableShadows.onChange().connect([this](bool val) { mShadowMap->setEnabled(val); });

  pEnableShadowsFreeze.onChange().connect([this](bool val) { mShadowMap->setFreezeCascades(val); });

  pShadowMapResolution.onChange().connect(
      [this](int val) { mShadowMap->setResolution((uint32_t)val); });

  pShadowMapCascades.onChange().connect([this](int) { calculateCascades(); });

  pShadowMapBias.onChange().connect([this](float val) { mShadowMap->setBias(val * 0.0001f); });

  pShadowMapSplitDistribution.onChange().connect([this](float) { calculateCascades(); });

  pShadowMapRange.onChange().connect([this](glm::vec2) { calculateCascades(); });

  pShadowMapExtension.onChange().connect([this](glm::vec2) { calculateCascades(); });

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
  mToneMappingNode       = std::make_shared<graphics::ToneMappingNode>(mHDRBuffer, true);
  auto toneMappingGLNode = pSG->NewOpenGLNode(pSG->GetRoot(), mToneMappingNode.get());
  VistaOpenSGMaterialTools::SetSortKeyOnSubtree(
      toneMappingGLNode, static_cast<int>(utils::DrawOrder::eToneMapping));

  pGlowIntensity.onChange().connect([this](float val) { mToneMappingNode->setGlowIntensity(val); });

  pExposureCompensation.onChange().connect(
      [this](float val) { mToneMappingNode->setExposureCompensation(val); });

  pExposureAdaptionSpeed.onChange().connect(
      [this](float val) { mToneMappingNode->setExposureAdaptionSpeed(val); });

  pAutoExposureRange.onChange().connect([this](glm::vec2 val) {
    mToneMappingNode->setMinAutoExposure(val[0]);
    mToneMappingNode->setMaxAutoExposure(val[1]);
  });

  pEnableHDR.onChange().connect([clearGLNode, toneMappingGLNode](bool enabled) {
    clearGLNode->SetIsEnabled(enabled);
    toneMappingGLNode->SetIsEnabled(enabled);
  });

  pEnableAutoExposure.onChange().connect(
      [this](bool enabled) { mToneMappingNode->setEnableAutoExposure(enabled); });

  pExposure.onChange().connect([this](float value) {
    if (!pEnableAutoExposure.get()) {
      mToneMappingNode->setExposure(value);
    }

    // Whenever the exposure changes, and if auto-glow is enabled, we change the glow intensity
    // based on the exposure value. The auto-glow amount is based on the current exposure relative
    // to the auto-exposure range.
    if (pEnableAutoGlow.get()) {
      float glow = (pAutoExposureRange.get()[0] - value) /
                   (pAutoExposureRange.get()[0] - pAutoExposureRange.get()[1]);
      pGlowIntensity = std::clamp(glow * 0.5f, 0.001f, 1.f);
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
  if (mSettings->mEnableSensorSizeControl) {
    VistaViewport* pViewport(GetVistaSystem()->GetDisplayManager()->GetViewports().begin()->second);
    int            sizeX = 0;
    int            sizeY = 0;
    pViewport->GetViewportProperties()->GetSize(sizeX, sizeY);
    float aspect = 1.f * sizeX / sizeY;

    VistaProjection::VistaProjectionProperties* pProjProps =
        pViewport->GetProjection()->GetProjectionProperties();

    float height = pSensorDiagonal.get() / std::sqrt(std::pow(aspect, 2.f) + 1.f);
    height /= pFocalLength.get();
    float width = aspect * height;
    pProjProps->SetProjPlaneExtents(-width / 2, width / 2, -height / 2, height / 2);
    pProjProps->SetProjPlaneMidpoint(0, 0, -1);
  }

  // Update exposure. If auto exposure is enabled, the property will reflect the exposure chosen by
  // the tonemapping node.
  if (pEnableAutoExposure.get()) {
    pExposure = mToneMappingNode->getExposure();
  }

  if (pEnableHDR.get()) {
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
  float              nearEnd = pShadowMapRange.get().x;
  float              farEnd  = pShadowMapRange.get().y;
  int                count   = pShadowMapCascades.get();
  std::vector<float> splits(count + 1);
  for (int i(0); i < splits.size(); ++i) {
    float alpha = (float)(i) / count;
    alpha       = std::pow(alpha, pShadowMapSplitDistribution.get());
    splits[i]   = glm::mix(nearEnd, farEnd, alpha);
  }
  mShadowMap->setCascadeSplits(splits);
  mShadowMap->setSunNearClipOffset(pShadowMapExtension.get().x);
  mShadowMap->setSunFarClipOffset(pShadowMapExtension.get().y);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cs::core
