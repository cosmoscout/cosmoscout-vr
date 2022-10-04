////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "Satellite.hpp"

#include <VistaKernel/GraphicsManager/VistaNodeBridge.h>
#include <VistaKernel/GraphicsManager/VistaTransformNode.h>
#include <VistaKernel/VistaSystem.h>
#include <VistaKernelOpenSGExt/VistaOpenSGMaterialTools.h>

#include "../../../src/cs-core/Settings.hpp"
#include "../../../src/cs-core/SolarSystem.hpp"
#include "../../../src/cs-graphics/GltfLoader.hpp"
#include "../../../src/cs-utils/convert.hpp"
#include "../../../src/cs-utils/utils.hpp"

#include <glm/gtc/type_ptr.hpp>
#include <utility>

namespace csp::satellites {

////////////////////////////////////////////////////////////////////////////////////////////////////

Satellite::Satellite(Plugin::Settings::Satellite const& config, std::string objectName,
    VistaSceneGraph* sceneGraph, std::shared_ptr<cs::core::Settings> settings,
    std::shared_ptr<cs::core::SolarSystem> solarSystem)
    : mSceneGraph(sceneGraph)
    , mSettings(std::move(settings))
    , mSolarSystem(std::move(solarSystem))
    , mModel(std::make_unique<cs::graphics::GltfLoader>(config.mModelFile, config.mEnvironmentMap))
    , mObjectName(std::move(objectName)) {

  mModel->setLightIntensity(15.0);
  mModel->setIBLIntensity(1.5);
  mModel->setLightColor(1.0, 1.0, 1.0);

  mAnchor.reset(sceneGraph->NewTransformNode(sceneGraph->GetRoot()));

  mModel->attachTo(sceneGraph, mAnchor.get());

  VistaOpenSGMaterialTools::SetSortKeyOnSubtree(
      mAnchor.get(), static_cast<int>(cs::utils::DrawOrder::eOpaqueItems));

  mAnchor->SetIsEnabled(false);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

Satellite::~Satellite() {
  mSceneGraph->GetRoot()->DisconnectChild(mAnchor.get());
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Satellite::update() {

  auto object  = mSolarSystem->getObject(mObjectName);
  bool visible = object && object->getIsBodyVisible();

  mAnchor->SetIsEnabled(visible);

  if (visible) {
    auto const& transform = object->getObserverRelativeTransform();
    mAnchor->SetTransform(glm::value_ptr(transform), true);

    float sunIlluminance(1.F);

    auto sunDirection = glm::vec3(mSolarSystem->getSunDirection(transform[3]));

    mModel->setLightDirection(sunDirection.x, sunDirection.y, sunDirection.z);

    if (mSettings->mGraphics.pEnableHDR.get()) {
      mModel->setEnableHDR(true);
      sunIlluminance = static_cast<float>(mSolarSystem->getSunIlluminance(transform[3]));
    }
    mModel->setLightIntensity(sunIlluminance);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::satellites
