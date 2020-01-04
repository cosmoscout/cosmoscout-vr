////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "GraphicsEngine.hpp"

#include <VistaKernel/GraphicsManager/VistaSceneGraph.h>
#include <VistaKernel/VistaSystem.h>
#include <spdlog/spdlog.h>

namespace cs::core {

////////////////////////////////////////////////////////////////////////////////////////////////////

GraphicsEngine::GraphicsEngine(std::shared_ptr<const core::Settings> const& settings) {

  // Tell the user what's going on.
  spdlog::debug("Creating GraphicsEngine.");

  auto pSG = GetVistaSystem()->GetGraphicsManager()->GetSceneGraph();

  pWidgetScale = settings->mWidgetScale;

  mShadowMap.setEnabled(false);
  mShadowMap.setResolution((uint32_t)pShadowMapResolution.get());
  mShadowMap.setBias(pShadowMapBias.get() * 0.0001f);
  pSG->NewOpenGLNode(pSG->GetRoot(), &mShadowMap);

  calculateCascades();

  pEnableShadows.onChange().connect([this](bool val) { mShadowMap.setEnabled(val); });

  pEnableShadowsFreeze.onChange().connect([this](bool val) { mShadowMap.setFreezeCascades(val); });

  pShadowMapResolution.onChange().connect(
      [this](int val) { mShadowMap.setResolution((uint32_t)val); });

  pShadowMapCascades.onChange().connect([this](int) { calculateCascades(); });

  pShadowMapBias.onChange().connect([this](float val) { mShadowMap.setBias(val * 0.0001f); });

  pShadowMapSplitDistribution.onChange().connect([this](float) { calculateCascades(); });

  pShadowMapRange.onChange().connect([this](glm::vec2) { calculateCascades(); });
  pShadowMapExtension.onChange().connect([this](glm::vec2) { calculateCascades(); });
}

////////////////////////////////////////////////////////////////////////////////////////////////////

GraphicsEngine::~GraphicsEngine() {
  // Tell the user what's going on.
  spdlog::debug("Deleting GraphicsEngine.");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void GraphicsEngine::registerCaster(graphics::ShadowCaster* caster) {
  mShadowMap.registerCaster(caster);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void GraphicsEngine::unregisterCaster(graphics::ShadowCaster* caster) {
  mShadowMap.deregisterCaster(caster);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void GraphicsEngine::setSunDirection(glm::vec3 const& direction) {
  mShadowMap.setSunDirection(VistaVector3D(direction.x, direction.y, direction.z));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

graphics::ShadowMap const* GraphicsEngine::getShadowMap() const {
  return &mShadowMap;
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
  mShadowMap.setCascadeSplits(splits);
  mShadowMap.setSunNearClipOffset(pShadowMapExtension.get().x);
  mShadowMap.setSunFarClipOffset(pShadowMapExtension.get().y);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cs::core
