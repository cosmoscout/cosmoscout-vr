////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "MultiPointTool.hpp"

#include <utility>

#include "../../cs-core/SolarSystem.hpp"
#include "../../cs-scene/CelestialSurface.hpp"
#include "../../cs-utils/convert.hpp"
#include "../InputManager.hpp"

namespace cs::core::tools {

////////////////////////////////////////////////////////////////////////////////////////////////////

MultiPointTool::MultiPointTool(std::shared_ptr<InputManager> pInputManager,
    std::shared_ptr<SolarSystem> pSolarSystem, std::shared_ptr<Settings> settings,
    std::string objectName)
    : Tool(std::move(objectName))
    , mInputManager(std::move(pInputManager))
    , mSolarSystem(std::move(pSolarSystem))
    , mSettings(std::move(settings)) {

  // If pAddPointMode is true, a new point will be added on a left mouse button click.
  mLeftButtonConnection = mInputManager->pButtons[0].connect([this](bool pressed) {
    if (pAddPointMode.get() && !pressed) {
      auto planet = mInputManager->pHoveredObject.get().mObject;
      if (planet) {
        addPoint();
      }
    }
  });

  // If pAddPointMode is true, it will be set to false on a right mouse button click
  // the point which was currently added will be removed again.
  mRightButtonConnection = mInputManager->pButtons[1].connect([this](bool pressed) {
    if (pAddPointMode.get() && !pressed) {
      pAddPointMode = false;
      onPointRemoved(static_cast<int32_t>(mPoints.size() - 1));
      mPoints.pop_back();
    }
  });
}

////////////////////////////////////////////////////////////////////////////////////////////////////

MultiPointTool::~MultiPointTool() {
  // Disconnect the mouse button slots.
  mInputManager->pButtons[0].disconnect(mLeftButtonConnection);
  mInputManager->pButtons[1].disconnect(mRightButtonConnection);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void MultiPointTool::addPoint(std::optional<glm::dvec2> const& lngLat) {
  // Add the Mark to the list.
  mPoints.emplace_back(
      std::make_shared<DeletableMark>(mInputManager, mSolarSystem, mSettings, getObjectName()));

  // if there is a planet intersection, move the point to the intersection location
  if (lngLat) {
    mPoints.back()->pLngLat = lngLat.value();
  } else {
    auto intersection = mInputManager->pHoveredObject.get();
    if (intersection.mObject && intersection.mObjectName == getObjectName()) {
      auto       radii = intersection.mObject->getRadii();
      glm::dvec2 pos   = cs::utils::convert::cartesianToLngLat(intersection.mPosition, radii);
      mPoints.back()->pLngLat = pos;
    }
  }

  // register callback to update line vertices when the landmark position has been changed
  mPoints.back()->pLngLat.connect([this](glm::dvec2 const& /*unused*/) { onPointMoved(); });

  // Update the color.
  mPoints.back()->pColor.connectFrom(pColor);

  // Update scaling distance.
  mPoints.back()->pScaleDistance.connectFrom(pScaleDistance);

  // Call update once since new data is available.
  onPointAdded();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void MultiPointTool::update() {
  bool anyPointSelected = false;
  int  index            = 0;

  // Update all points and remove them if required.
  for (auto mark = mPoints.begin(); mark != mPoints.end();) {
    if ((*mark)->pShouldDelete.get()) {
      onPointRemoved(index);
      mark = mPoints.erase(mark);
    } else {
      anyPointSelected |= static_cast<bool>((*mark)->pSelected.get());
      (*mark)->update();
      ++mark;
      ++index;
    }
  }

  pAnyPointSelected = anyPointSelected;

  // Request deletion of the tool itself if there are no points left.
  if (mPoints.empty()) {
    pShouldDelete = true;
  }

  // If pAddPointMode is true, move the last point to the current body
  // intersection position (if there is any).
  if (pAddPointMode.get()) {
    auto intersection = mInputManager->pHoveredObject.get();
    if (intersection.mObject) {
      auto radii = intersection.mObject->getRadii();
      mPoints.back()->pLngLat =
          cs::utils::convert::cartesianToLngLat(intersection.mPosition, radii);
    }
  }

  // This seems to be the first time the tool is moved, so we have to store the distance to the
  // observer so that we can scale the tool later based on the observer's position.
  if (pScaleDistance.get() < 0) {
    auto object = mSolarSystem->getObject(getObjectName());

    pScaleDistance =
        mSolarSystem->getObserver().getScale() *
        glm::length(object->getObserverRelativePosition(mPoints.back()->getPosition()));
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void MultiPointTool::setObjectName(std::string name) {
  Tool::setObjectName(std::move(name));

  for (auto const& point : mPoints) {
    point->setObjectName(getObjectName());
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::vector<glm::dvec2> MultiPointTool::getPositions() const {
  std::vector<glm::dvec2> result;
  result.reserve(mPoints.size());

  for (auto const& p : mPoints) {
    result.emplace_back(p->pLngLat.get());
  }

  return result;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void MultiPointTool::setPositions(std::vector<glm::dvec2> const& positions) {

  // We try to re-use the DeletableMarks we already have in order to make this faster. If we have
  // more than required, delete the rest.
  while (mPoints.size() > positions.size()) {
    onPointRemoved(static_cast<int32_t>(mPoints.size() - 1));
    mPoints.pop_back();
  }

  // Now update the positions of all the points we still have and then add new ones as required.
  auto it = mPoints.begin();
  for (size_t i(0); i < positions.size(); ++i, ++it) {
    if (i < mPoints.size()) {
      if ((*it)->pLngLat.get() != positions[i]) {
        (*it)->pLngLat = positions[i];
        onPointMoved();
      }
    } else {
      addPoint(positions[i]);
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cs::core::tools
