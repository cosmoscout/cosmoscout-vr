////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "MultiPointTool.hpp"

#include <utility>

#include "../../cs-scene/CelestialBody.hpp"
#include "../../cs-utils/convert.hpp"
#include "../InputManager.hpp"

namespace cs::core::tools {

////////////////////////////////////////////////////////////////////////////////////////////////////

MultiPointTool::MultiPointTool(std::shared_ptr<InputManager> pInputManager,
    std::shared_ptr<SolarSystem> pSolarSystem, std::shared_ptr<GraphicsEngine> graphicsEngine,
    std::shared_ptr<TimeControl> pTimeControl, std::string sCenter, std::string sFrame)
    : mInputManager(std::move(pInputManager))
    , mSolarSystem(std::move(pSolarSystem))
    , mGraphicsEngine(std::move(graphicsEngine))
    , mTimeControl(std::move(pTimeControl))
    , mCenter(std::move(sCenter))
    , mFrame(std::move(sFrame)) {

  // if pAddPointMode is true, a new point will be add on left mouse button click
  mLeftButtonConnection = mInputManager->pButtons[0].connect([this](bool pressed) {
    if (pAddPointMode.get() && !pressed) {
      auto planet = mInputManager->pHoveredObject.get().mObject;
      if (planet) {
        addPoint();
      }
    }
  });

  // if pAddPointMode is true, it will be set to false on right mouse button click
  // the point which was currently added will be removed again
  mRightButtonConnection = mInputManager->pButtons[1].connect([this](bool pressed) {
    if (pAddPointMode.get() && !pressed) {
      pAddPointMode = false;
      mPoints.pop_back();
      onPointRemoved(static_cast<int32_t>(mPoints.size()));
    }
  });
}

////////////////////////////////////////////////////////////////////////////////////////////////////

MultiPointTool::~MultiPointTool() {
  // disconnect the mouse button slots
  mInputManager->pButtons[0].disconnect(mLeftButtonConnection);
  mInputManager->pButtons[1].disconnect(mRightButtonConnection);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void MultiPointTool::addPoint() {
  // add the Mark to the list
  mPoints.emplace_back(std::make_shared<DeletableMark>(
      mInputManager, mSolarSystem, mGraphicsEngine, mTimeControl, mCenter, mFrame));

  // if there is a planet intersection, move the point to the intersection location
  auto intersection = mInputManager->pHoveredObject.get();
  if (intersection.mObject) {
    auto body = std::dynamic_pointer_cast<cs::scene::CelestialBody>(intersection.mObject);

    if (body) {
      auto       radii = body->getRadii();
      glm::dvec2 pos =
          cs::utils::convert::toLngLatHeight(intersection.mPosition, radii[0], radii[0]).xy();
      mPoints.back()->pLngLat = pos;
    }
  }

  // register callback to update line vertices when the landmark position has been changed
  mPoints.back()->pLngLat.connect([this](glm::dvec2 const&) { onPointMoved(); });

  // call update once since new data is available
  onPointAdded();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void MultiPointTool::update() {
  bool anyPointSelected = false;
  int  index            = 0;

  // update all points and remove them if required
  for (auto mark = mPoints.begin(); mark != mPoints.end();) {
    if ((*mark)->pShouldDelete.get()) {
      mark = mPoints.erase(mark);
      onPointRemoved(index);
    } else {
      anyPointSelected |= static_cast<bool>((*mark)->pSelected.get());
      (*mark)->update();
      ++mark;
      ++index;
    }
  }

  pAnyPointSelected = anyPointSelected;

  // request deletion of the tool itself if there are no points left
  if (mPoints.empty()) {
    pShouldDelete = true;
  }

  // if pAddPointMode is true, move the last point to the current planet
  // intersection position (if there is any)
  if (pAddPointMode.get()) {
    auto intersection = mInputManager->pHoveredObject.get();
    if (intersection.mObject) {
      auto body = std::dynamic_pointer_cast<cs::scene::CelestialBody>(intersection.mObject);

      if (body) {
        auto radii = body->getRadii();
        mPoints.back()->pLngLat =
            cs::utils::convert::toLngLatHeight(intersection.mPosition, radii[0], radii[0]).xy();
      }
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::string const& MultiPointTool::getCenterName() const {
  return mCenter;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::string const& MultiPointTool::getFrameName() const {
  return mFrame;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cs::core::tools
