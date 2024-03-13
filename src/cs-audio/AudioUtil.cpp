////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "AudioUtil.hpp"
#include "../cs-scene/CelestialAnchor.hpp"
#include "../cs-scene/CelestialSurface.hpp"
#include "../cs-utils/convert.hpp"
#include "logger.hpp"
#include <cmath>
#include <iostream>

namespace cs::audio {

double AudioUtil::getObserverScaleAt(
    glm::dvec3 position, double ObserverScale, std::shared_ptr<cs::core::Settings> settings) {

  // First we have to find the planet which is closest to the position.
  std::shared_ptr<const scene::CelestialObject> closestObject;
  double dClosestDistance = std::numeric_limits<double>::max();

  // Here we will store the position of the source relative to the closestObject.
  glm::dvec3 vClosestPlanetPosition(0.0);

  for (auto const& [name, object] : settings->mObjects) {

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

    glm::dvec3 vSourcePosToObject(vObjectPosToObserver.x - position.x,
        vObjectPosToObserver.y - position.y, vObjectPosToObserver.z - position.z);
    double     dDistance = glm::length(vSourcePosToObject) - radii[0];

    if (dDistance < dClosestDistance) {
      closestObject          = object;
      dClosestDistance       = dDistance;
      vClosestPlanetPosition = vSourcePosToObject;
    }
  }

  // Now that we found a closest body, we will scale the  in such a way, that the closest
  // body is rendered at a distance between settings->mSceneScale.mCloseVisualDistance and
  // settings->mSceneScale.mFarVisualDistance (in meters).
  if (closestObject) {

    // First we calculate the *real* world-space distance to the planet (incorporating surface
    // elevation).
    auto radii        = closestObject->getRadii() * closestObject->getScale();
    auto lngLatHeight = cs::utils::convert::cartesianToLngLatHeight(vClosestPlanetPosition, radii);
    double dRealDistance = lngLatHeight.z;

    if (closestObject->getSurface()) {
      dRealDistance -= closestObject->getSurface()->getHeight(lngLatHeight.xy()) *
                       settings->mGraphics.pHeightScale.get();
    }

    if (std::isnan(dRealDistance)) {
      return -1.0;
    }

    // The render distance between settings->mSceneScale.mCloseVisualDistance and
    // settings->mSceneScale.mFarVisualDistance is chosen based on the observer's world-space
    // distance between settings->mSceneScale.mFarRealDistance and
    // settings->mSceneScale.mCloseRealDistance (also in meters).
    double interpolate = 1.0;

    if (settings->mSceneScale.mFarRealDistance != settings->mSceneScale.mCloseRealDistance) {
      interpolate = glm::clamp(
          (dRealDistance - settings->mSceneScale.mCloseRealDistance) /
              (settings->mSceneScale.mFarRealDistance - settings->mSceneScale.mCloseRealDistance),
          0.0, 1.0);
    }

    double dScale = dRealDistance / glm::mix(settings->mSceneScale.mCloseVisualDistance,
                                        settings->mSceneScale.mFarVisualDistance, interpolate);
    dScale = glm::clamp(dScale, settings->mSceneScale.mMinScale, settings->mSceneScale.mMaxScale);

    return dScale;
  }
  return -1.0;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void AudioUtil::printAudioSettings(std::shared_ptr<std::map<std::string, std::any>> map) {
  for (auto [key, val] : (*map)) {

    std::cout << key << ": ";

    if (val.type() == typeid(int)) {
      std::cout << std::any_cast<int>(val) << std::endl;
      continue;
    }

    if (val.type() == typeid(bool)) {
      std::cout << (std::any_cast<bool>(val) ? "true" : "false") << std::endl;
      continue;
    }

    if (val.type() == typeid(float)) {
      std::cout << std::any_cast<float>(val) << std::endl;
      continue;
    }

    if (val.type() == typeid(std::string)) {
      std::cout << std::any_cast<std::string>(val) << std::endl;
      continue;
    }

    if (val.type() == typeid(glm::dvec3)) {
      auto v3 = std::any_cast<glm::dvec3>(val);
      std::cout << v3.x << ", " << v3.y << ", " << v3.z << std::endl;
      continue;
    }

    std::cout << "type not yet supported for printing in AudioUtil::printAudioSettings()"
              << std::endl;
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void AudioUtil::printAudioSettings(
    const std::shared_ptr<const std::map<std::string, std::any>> map) {
  printAudioSettings(std::const_pointer_cast<std::map<std::string, std::any>>(map));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

double inverseClamped(
    double distance, ALfloat rollOffFactor, ALfloat referenceDistance, ALfloat maxDistance) {
  return referenceDistance / (referenceDistance + rollOffFactor * (distance - referenceDistance));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

double linearClamped(
    double distance, ALfloat rollOffFactor, ALfloat referenceDistance, ALfloat maxDistance) {
  return (1 - rollOffFactor * (distance - referenceDistance) / (maxDistance - referenceDistance));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

double exponentClamped(
    double distance, ALfloat rollOffFactor, ALfloat referenceDistance, ALfloat maxDistance) {
  return std::pow((distance / referenceDistance), -1 * rollOffFactor);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

ALfloat AudioUtil::computeFallOffFactor(double distance, std::string model, ALfloat fallOffStart, ALfloat fallOffEnd) {

  if (distance < fallOffStart) {
    logger().warn("AudioUtil::setFallOffFactor: distance cannot be smaller then the sources fallOffStart distance!");
    return -1.f;
  }

  if (distance > fallOffEnd) {
    logger().warn("AudioUtil::setFallOffFactor: distance cannot be larger then the sources fallOffEnd distance!");
    return -1.f;
  }
  
  // Get function for distance calculation
  double (*distanceFunction)(double, ALfloat, ALfloat, ALfloat);

  if (model == "inverse") {
    distanceFunction = inverseClamped;

  } else if (model == "linear") {
    distanceFunction = linearClamped;

  } else if (model == "exponent") {
    distanceFunction = exponentClamped;

  } else {
    logger().warn("AudioUtil::setFallOffFactor: Invalid distance model!");
    return -1.f;
  }
  
  // Compute FallOffFactor
  ALfloat fallOffFactor = 0.01f;
  float stepSize = 0.1f;
  float resultWindow = 0.001f;

  while (distanceFunction(distance, fallOffFactor, fallOffStart, fallOffEnd) > resultWindow) {
    fallOffFactor += stepSize;
  }

  return fallOffFactor;
}

} // namespace cs::audio