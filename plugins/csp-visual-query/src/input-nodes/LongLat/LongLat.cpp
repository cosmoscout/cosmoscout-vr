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

namespace {
double getNow() {
  auto time        = std::chrono::system_clock::now();
  auto since_epoch = time.time_since_epoch();
  return std::chrono::duration_cast<std::chrono::microseconds>(since_epoch).count() * 1e-6;
}

} // namespace

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

      // Whenever the user clicked on the surface, we write create a new Mark object at this
      // position.
      mMark =
          std::make_unique<csl::tools::Mark>(mInputManager, mSolarSystem, mSettings, objectName);
      mMark->pLngLat = cs::utils::convert::toRadians(
          glm::dvec2(lngLat.x, lngLat.y));        // Convert to radians for the Mark object.
      mMark->pColor = glm::vec3(0.75, 1.0, 0.75); // Set the color of the mark.

      mMark->pLngLat.connectAndTouch([this](glm::dvec2 const& lngLat) {
        mValue = {cs::utils::convert::toDegrees(lngLat.x), cs::utils::convert::toDegrees(lngLat.y)};
        mLastUpdateTime = getNow();
      });
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
  std::cout << "LongLat::process() called" << std::endl;
  writeOutput("value", mValue);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void LongLat::tick() {
  if (mMark) {
    mMark->update();

    if (mLastUpdateTime > 0) {
      double now = getNow();

      if (now - mLastUpdateTime > 0.5) {
        process();
        mLastUpdateTime = 0;
      }
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void LongLat::onMessageFromJS(nlohmann::json const& message) {
  mWaitingForClick = true;
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
