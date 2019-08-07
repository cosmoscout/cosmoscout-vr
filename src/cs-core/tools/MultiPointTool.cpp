////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "MultiPointTool.hpp"

#include "../../cs-scene/CelestialBody.hpp"
#include "../../cs-utils/convert.hpp"
#include "../InputManager.hpp"

namespace cs::core::tools {

////////////////////////////////////////////////////////////////////////////////////////////////////

MultiPointTool::MultiPointTool(std::shared_ptr<InputManager> const& pInputManager,
    std::shared_ptr<SolarSystem> const&                             pSolarSystem,
    std::shared_ptr<GraphicsEngine> const&                          graphicsEngine,
    std::shared_ptr<GuiManager> const&                              pGuiManager,
    std::shared_ptr<TimeControl> const& pTimeControl, std::string const& sCenter,
    std::string const& sFrame)
    : mInputManager(pInputManager)
    , mSolarSystem(pSolarSystem)
    , mGraphicsEngine(graphicsEngine)
    , mGuiManager(pGuiManager)
    , mTimeControl(pTimeControl)
    , mCenter(sCenter)
    , mFrame(sFrame) {

  // if pAddPointMode is true, a new point will be add on left mouse button click
  mLeftButtonConnection = mInputManager->pButtons[0].onChange().connect([this](bool pressed) {
    if (pAddPointMode.get() && !pressed) {
      auto planet = mInputManager->pHoveredObject.get().mObject;
      if (planet) {
        addPoint();
      }
    }
  });

  // if pAddPointMode is true, it will be set to false on right mouse button click
  // the point which was currently added will be removed again
  mRightButtonConnection = mInputManager->pButtons[1].onChange().connect([this](bool pressed) {
    if (pAddPointMode.get() && !pressed) {
      pAddPointMode = false;
      mPoints.pop_back();
      onPointRemoved();
    }
  });
}

////////////////////////////////////////////////////////////////////////////////////////////////////

MultiPointTool::~MultiPointTool() {
  // disconnect the mouse button slots
  mInputManager->pButtons[0].onChange().disconnect(mLeftButtonConnection);
  mInputManager->pButtons[1].onChange().disconnect(mRightButtonConnection);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void MultiPointTool::addPoint() {
  // add the Mark to the list
  mPoints.emplace_back(std::make_shared<DeletableMark>(
      mInputManager, mSolarSystem, mGraphicsEngine, mGuiManager, mTimeControl, mCenter, mFrame));

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
  mPoints.back()->pLngLat.onChange().connect([this](glm::dvec2 const& lngLat) { onPointMoved(); });

  // call update once since new data is available
  onPointAdded();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void MultiPointTool::update() {
  bool anyPointSelected = false;

  // update all points and remove them if required
  for (auto mark = mPoints.begin(); mark != mPoints.end();) {
    if ((*mark)->pShouldDelete.get()) {
      mark = mPoints.erase(mark);
      onPointRemoved();
    } else {
      anyPointSelected |= (*mark)->pSelected.get();
      (*mark)->update();
      ++mark;
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

} // namespace cs::core::tools
