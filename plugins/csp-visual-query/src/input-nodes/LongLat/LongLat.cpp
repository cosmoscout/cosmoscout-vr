////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "LongLat.hpp"

#include "../../../../../src/cs-core/InputManager.hpp"
#include "../../../../../src/cs-utils/convert.hpp"
#include "../../../../../src/cs-utils/filesystem.hpp"

#include <iostream>

namespace csp::visualquery {

////////////////////////////////////////////////////////////////////////////////////////////////////

const std::string LongLat::sName = "LongLat";

////////////////////////////////////////////////////////////////////////////////////////////////////

std::string LongLat::sSource() {
  return cs::utils::filesystem::loadToString(
      "../share/resources/nodes/csp-visual-query/LongLat.js");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::unique_ptr<LongLat> LongLat::sCreate(std::shared_ptr<cs::core::InputManager> inputManager,
    std::shared_ptr<cs::core::SolarSystem>                                        solarSystem,
    std::shared_ptr<cs::core::Settings>                                           settings) {
  return std::make_unique<LongLat>(
      std::move(inputManager), std::move(solarSystem), std::move(settings));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

LongLat::LongLat(std::shared_ptr<cs::core::InputManager> inputManager,
    std::shared_ptr<cs::core::SolarSystem>               solarSystem,
    std::shared_ptr<cs::core::Settings>                  settings)
    : mInputManager(std::move(inputManager))
    , mSolarSystem(std::move(solarSystem))
    , mSettings(std::move(settings))
    , mValue(0.0, 0.0) {

  mOnClickConnection = mInputManager->pButtons[0].connect([this](bool pressed) {
    if (!pressed && mWaitingForClick && !mInputManager->pHoveredGuiItem.get()) {
      auto object     = mInputManager->pHoveredObject.get().mObject;
      auto objectName = mInputManager->pHoveredObject.get().mObjectName;

      if (!object) {
        return;
      }

      auto radii  = object->getRadii();
      auto lngLat = cs::utils::convert::toDegrees(cs::utils::convert::cartesianToLngLat(
          mInputManager->pHoveredObject.get().mPosition, radii));

      mWaitingForClick = false;

      std::cout << "Clicked on " << objectName << ": " << lngLat.x << ", " << lngLat.y << std::endl;

      // Whenever the user clicked on the surface, we write create a new Mark object at this
      // position.
      mMark =
          std::make_unique<csl::tools::Mark>(mInputManager, mSolarSystem, mSettings, objectName);
      mMark->pLngLat = cs::utils::convert::toRadians(
          glm::dvec2(lngLat.x, lngLat.y));        // Convert to radians for the Mark object.
      mMark->pColor = glm::vec3(0.75, 1.0, 0.75); // Set the color of the mark.
    }
  });
}

////////////////////////////////////////////////////////////////////////////////////////////////////

LongLat::~LongLat() {
  mInputManager->pButtons[0].disconnect(mOnClickConnection);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::string const& LongLat::getName() const {
  return sName;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void LongLat::process() {
  writeOutput("value", mValue);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void LongLat::tick() {
  if (mMark) {
    std::cout << "Ticking mark" << std::endl;
    mMark->update();
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void LongLat::onMessageFromJS(nlohmann::json const& message) {
  // The message sent via CosmoScout.sendMessageToCPP() contains the selected number.
  std::cout << "button clicked" << std::endl;
  mWaitingForClick = true;

  // Whenever the user entered a number, we write it to the output socket by calling the process()
  // method. Writing the output will not trigger a graph reprocessing right away, it will only queue
  // up the connected nodes for being processed in the next update step (and only if the value
  // actually changed).
  process();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

nlohmann::json LongLat::getData() const {
  return {{"value", mValue}};
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void LongLat::setData(nlohmann::json const& json) {
  mValue = json["value"];
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::visualquery
