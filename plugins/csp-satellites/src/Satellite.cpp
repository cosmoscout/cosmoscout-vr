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
    , mSolarSystem(solarSystem)
    , mObjectName(objectName)
    , mViewPointer(std::make_unique<ViewPointer>(config, solarSystem, objectName)) {

  mAnchor.reset(sceneGraph->NewTransformNode(sceneGraph->GetRoot()));

  addModel("../share/resources/models/VLEO_centered.glb", config.mEnvironmentMap);
  addModel("../share/resources/models/VLEO_alt.glb", config.mEnvironmentMap);
  addModel("../share/resources/models/IdeatoOrbit-rev01.glb", config.mEnvironmentMap);
  addModel("../share/resources/models/IdeatoOrbit-rev01_double.glb", config.mEnvironmentMap);
  config.mModelFile.connectAndTouch([this](std::string modelFile) {
    for (auto& model : mModels) {
      model.second->setActive(model.first == modelFile);
    }
  });

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

    for (auto& model : mModels) {
      model.second->setLightDirection(sunDirection.x, sunDirection.y, sunDirection.z);

      if (mSettings->mGraphics.pEnableHDR.get()) {
        model.second->setEnableHDR(true);
        sunIlluminance = static_cast<float>(mSolarSystem->getSunIlluminance(transform[3]));
      }
      model.second->setLightIntensity(sunIlluminance);
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Satellite::addModel(std::string const& modelFile, std::string const& envMapFile) {
  mModels[modelFile] = std::make_unique<cs::graphics::GltfLoader>(modelFile, envMapFile);

  mModels.at(modelFile)->setLightIntensity(15.0);
  mModels.at(modelFile)->setIBLIntensity(1.5);
  mModels.at(modelFile)->setLightColor(1.0, 1.0, 1.0);

  mModels.at(modelFile)->attachTo(mSceneGraph, mAnchor.get());
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::satellites
