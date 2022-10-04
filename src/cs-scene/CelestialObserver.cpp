////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "CelestialObserver.hpp"

#include "logger.hpp"

namespace cs::scene {

////////////////////////////////////////////////////////////////////////////////////////////////////

CelestialObserver::CelestialObserver(std::string const& sCenterName, std::string const& sFrameName)
    : CelestialAnchor(sCenterName, sFrameName) {
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void CelestialObserver::updateMovementAnimation(double tTime) {
  if (mAnimationInProgress) {
    mPosition = mAnimatedPosition.get(tTime);
    mRotation = mAnimatedRotation.get(tTime);

    if (mAnimatedPosition.mEndTime < tTime) {
      mAnimationInProgress = false;
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void CelestialObserver::setPosition(glm::dvec3 const& vPos) {
  if (!mAnimationInProgress) {
    CelestialAnchor::setPosition(vPos);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void CelestialObserver::setRotation(glm::dquat const& qRot) {
  if (!mAnimationInProgress) {
    CelestialAnchor::setRotation(qRot);
  }
}
////////////////////////////////////////////////////////////////////////////////////////////////////

void CelestialObserver::changeOrigin(
    std::string const& sCenterName, std::string const& sFrameName, double dSimulationTime) {

  mAnimationInProgress = false;

  cs::scene::CelestialAnchor target(sCenterName, sFrameName);

  glm::dvec3 pos = target.getRelativePosition(dSimulationTime, *this);
  glm::dquat rot = target.getRelativeRotation(dSimulationTime, *this);

  setCenterName(sCenterName);
  setFrameName(sFrameName);

  setRotation(rot);
  setPosition(pos);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void CelestialObserver::moveTo(std::string const& sCenterName, std::string const& sFrameName,
    glm::dvec3 const& position, glm::dquat const& rotation, double dSimulationTime,
    double dRealStartTime, double dRealEndTime) {
  mAnimationInProgress = false;

  // Perform no animation at all if end time is not greater than start time.
  if (dRealStartTime >= dRealEndTime) {
    setCenterName(sCenterName);
    setFrameName(sFrameName);
    setRotation(rotation);
    setPosition(position);

  } else {
    cs::scene::CelestialAnchor target(sCenterName, sFrameName);

    try {
      glm::dvec3 startPos = target.getRelativePosition(dSimulationTime, *this);
      glm::dquat startRot = target.getRelativeRotation(dSimulationTime, *this);

      setCenterName(sCenterName);
      setFrameName(sFrameName);

      double cosTheta = glm::dot(startRot, rotation);

      // If cosTheta < 0, the interpolation will take the long way around the sphere.
      // To fix this, one quat must be negated.
      if (cosTheta < 0.0) {
        startRot = -startRot;
      }

      setRotation(startRot);
      setPosition(startPos);

      mAnimatedPosition = utils::AnimatedValue<glm::dvec3>(
          startPos, position, dRealStartTime, dRealEndTime, utils::AnimationDirection::eInOut);

      mAnimatedRotation = utils::AnimatedValue<glm::dquat>(
          startRot, rotation, dRealStartTime, dRealEndTime, utils::AnimationDirection::eInOut);

      mAnimationInProgress = true;
    } catch (std::exception const& e) {
      // Getting the relative transformation may fail due to insufficient SPICE data.
      logger().warn("CelestialObserver::moveTo failed: {}", e.what());
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool CelestialObserver::isAnimationInProgress() const {
  return mAnimationInProgress;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cs::scene
