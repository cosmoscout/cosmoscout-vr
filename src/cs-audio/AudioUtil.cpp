////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "AudioUtil.hpp"
#include "../logger.hpp"
#include "../../cs-scene/CelestialAnchor.hpp"
#include "../../cs-scene/CelestialSurface.hpp"
#include "../../cs-utils/convert.hpp"
#include <cmath>

namespace cs::audio {

double AudioUtil::getObserverScaleAt(glm::dvec3 position, double ObserverScale) {

  // First we have to find the planet which is closest to the position.
  std::shared_ptr<const scene::CelestialObject> closestObject;
  double dClosestDistance = std::numeric_limits<double>::max();

  // Here we will store the position of the source relative to the closestObject.
  glm::dvec3 vClosestPlanetPosition(0.0);

  for (auto const& [name, object] : mSettings->mObjects) {

    // Skip non-existent objects.
    if (!object->getIsInExistence() || !object->getHasValidPosition() ||
        !object->getIsTrackable()) {
      continue;
    }

    // Skip objects with an unknown radius.
    auto radii = object->getRadii() * object->getScale();
    if (radii.x <= 0.0 || radii.y <= 0.0 || radii.z <= 0.0) {
      continue;
    }

    // Finally check if the current body is closest to the source. We won't incorporate surface
    // elevation in this check.
    auto vObjectPosToObserver = object->getObserverRelativePosition();
    vObjectPosToObserver *= static_cast<float>(ObserverScale); 

    glm::dvec3 vSourcePosToObject(
      vObjectPosToObserver.x - sourcePosToObserver.x,
      vObjectPosToObserver.y - sourcePosToObserver.y,
      vObjectPosToObserver.z - sourcePosToObserver.z
    );
    double dDistance = glm::length(vSourcePosToObject) - radii[0];

    if (dDistance < dClosestDistance) {
      closestObject            = object;
      dClosestDistance         = dDistance;
      vClosestPlanetPosition   = vSourcePosToObject;
    }
  }

  // Now that we found a closest body, we will scale the  in such a way, that the closest
  // body is rendered at a distance between mSettings->mSceneScale.mCloseVisualDistance and
  // mSettings->mSceneScale.mFarVisualDistance (in meters).
  if (closestObject) {

    // First we calculate the *real* world-space distance to the planet (incorporating surface
    // elevation).
    auto radii = closestObject->getRadii() * closestObject->getScale();
    auto lngLatHeight =
        cs::utils::convert::cartesianToLngLatHeight(vClosestPlanetSourcePosition, radii);
    double dRealDistance = lngLatHeight.z;

    if (closestObject->getSurface()) {
      dRealDistance -= closestObject->getSurface()->getHeight(lngLatHeight.xy()) *
                       mSettings->mGraphics.pHeightScale.get();
    }

    if (std::isnan(dRealDistance)) {
      return -1.0;
    }

    // The render distance between mSettings->mSceneScale.mCloseVisualDistance and
    // mSettings->mSceneScale.mFarVisualDistance is chosen based on the observer's world-space
    // distance between mSettings->mSceneScale.mFarRealDistance and
    // mSettings->mSceneScale.mCloseRealDistance (also in meters).
    double interpolate = 1.0;

    if (mSettings->mSceneScale.mFarRealDistance != mSettings->mSceneScale.mCloseRealDistance) {
      interpolate = glm::clamp(
          (dRealDistance - mSettings->mSceneScale.mCloseRealDistance) /
              (mSettings->mSceneScale.mFarRealDistance - mSettings->mSceneScale.mCloseRealDistance),
          0.0, 1.0);
    }

    double dScale = dRealDistance / glm::mix(mSettings->mSceneScale.mCloseVisualDistance,
                                        mSettings->mSceneScale.mFarVisualDistance, interpolate);
    dScale = glm::clamp(dScale, mSettings->mSceneScale.mMinScale, mSettings->mSceneScale.mMaxScale);
    
    return dScale;
  }
  return -1.0;
}
////////////////////////////////////////////////////////////////////////////////////////////////////


} // namespace cs::audio