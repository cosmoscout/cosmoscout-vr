////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "VolumeRenderer.hpp"

#include "SinglePassRaycaster.hpp"

#include "../../../../src/cs-utils/filesystem.hpp"
#include "../../../src/cs-core/Settings.hpp"
#include "../../../src/cs-core/SolarSystem.hpp"
#include "../../types/types.hpp"

namespace csp::visualquery {

////////////////////////////////////////////////////////////////////////////////////////////////////

const std::string VolumeRenderer::sName = "VolumeRenderer";

////////////////////////////////////////////////////////////////////////////////////////////////////

std::string VolumeRenderer::sSource() {
  return cs::utils::filesystem::loadToString(
      "../share/resources/nodes/csp-visual-query/VolumeRenderer.js");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::unique_ptr<VolumeRenderer> VolumeRenderer::sCreate(
    std::shared_ptr<cs::core::SolarSystem> solarSystem,
    std::shared_ptr<cs::core::Settings>    settings) {
  return std::make_unique<VolumeRenderer>(std::move(solarSystem), std::move(settings));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::string const& VolumeRenderer::getName() const {
  return sName;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

VolumeRenderer::VolumeRenderer(std::shared_ptr<cs::core::SolarSystem> solarSystem,
    std::shared_ptr<cs::core::Settings>                             settings)
    : mSolarSystem(std::move(solarSystem))
    , mSettings(std::move(settings)) {
  mRenderer = std::make_unique<SinglePassRaycaster>(mSolarSystem, mSettings);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

VolumeRenderer::~VolumeRenderer() = default;

////////////////////////////////////////////////////////////////////////////////////////////////////

nlohmann::json VolumeRenderer::getData() const {
  nlohmann::json data;

  std::set<std::string> centerNames{};

  for (const auto& item : mSettings->mObjects) {
    centerNames.insert(item.second->getCenterName());
  }

  std::vector<std::string> list{centerNames.begin(), centerNames.end()};
  list.insert(list.begin(), "None");

  data["options"]      = list;
  data["selectedBody"] = mRenderer->getCenter();

  return data;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void VolumeRenderer::setData(nlohmann::json const& json) {
  if (json.find("selectedBody") != json.end()) {
    mRenderer->setCenter(json["selectedBody"]);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void VolumeRenderer::init() {
  std::set<std::string> centerNames{};

  for (const auto& item : mSettings->mObjects) {
    centerNames.insert(item.second->getCenterName());
  }

  std::vector<std::string> data{centerNames.begin(), centerNames.end()};
  data.insert(data.begin(), "None");

  sendMessageToJS(data);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void VolumeRenderer::onMessageFromJS(const nlohmann::json& message) {
  mRenderer->setCenter(message.at("text").get<std::string>());
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void VolumeRenderer::process() {
  auto input = readInput<std::shared_ptr<Volume3D>>("Volume3D", nullptr);
  if (input.get() != mVolume.get()) {
    mRenderer->setData(input);
    mVolume = input;
  }

  auto lut = readInput<std::vector<glm::vec4>>("lut", {});
  mRenderer->setLUT(lut);
}

} // namespace csp::visualquery