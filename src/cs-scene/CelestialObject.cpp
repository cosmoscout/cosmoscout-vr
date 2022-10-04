////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "CelestialObject.hpp"

#include "../cs-utils/convert.hpp"
#include "CelestialObserver.hpp"
#include "logger.hpp"

#include <cspice/SpiceUsr.h>
#include <glm/gtc/type_ptr.hpp>

namespace cs::scene {

////////////////////////////////////////////////////////////////////////////////////////////////////

CelestialObject::CelestialObject(std::string sCenterName, std::string sFrameName)
    : CelestialAnchor(sCenterName, sFrameName) {
}

////////////////////////////////////////////////////////////////////////////////////////////////////

glm::dvec2 CelestialObject::getExistence() const {
  if (!mExistence && mExistenceAsStrings) {
    mExistence = glm::dvec2(utils::convert::time::toSpice(mExistenceAsStrings.value()[0]),
        utils::convert::time::toSpice(mExistenceAsStrings.value()[1]));
  }

  return mExistence.value_or(glm::dvec2(0.0));
}

void CelestialObject::setExistence(glm::dvec2 value) {
  mExistence          = value;
  mExistenceAsStrings = std::nullopt;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::array<std::string, 2> CelestialObject::getExistenceAsStrings() const {
  if (!mExistenceAsStrings && mExistence) {
    mExistenceAsStrings = {utils::convert::time::toString(mExistence.value()[0]),
        utils::convert::time::toString(mExistence.value()[1])};
  }

  return mExistenceAsStrings.value_or(
      std::array<std::string, 2>{"1950-01-02 00:00:00.000", "1950-01-02 00:00:00.000"});
}

void CelestialObject::setExistenceAsStrings(std::array<std::string, 2> const& value) {
  mExistence          = std::nullopt;
  mExistenceAsStrings = value;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

glm::dvec3 const& CelestialObject::getRadii() const {

  // If no radii were given to the object, we try once to get the from SPICE.
  if (mRadii == glm::dvec3(0.0) && mRadiiFromSPICE == glm::dvec3(-1.0)) {
    // get target id code
    SpiceInt     id{};
    SpiceBoolean found{};
    bodn2c_c(mCenterName.c_str(), &id, &found);

    // check if radius information is available
    if (!found || !bodfnd_c(id, "RADII")) {
      mRadiiFromSPICE = glm::dvec3(0.0);
    }

    // compute radius and convert it to meters
    SpiceInt   n{};
    glm::dvec3 result;
    bodvrd_c(mCenterName.c_str(), "RADII", 3, &n, glm::value_ptr(result));
    double const kmToMeter = 1000.0;
    result                 = result * kmToMeter;

    if (failed_c()) {
      std::array<SpiceChar, 320> msg{};
      getmsg_c("LONG", 320, msg.data());
      reset_c();
      logger().warn("Failed to retrieve SPICE radii for object {}: {}", mCenterName, msg.data());
      mRadiiFromSPICE = glm::dvec3(0.0);

    } else {

      // SPICE coordinates are different.
      mRadiiFromSPICE = glm::dvec3(result[1], result[2], result[0]);
    }

    return mRadiiFromSPICE;

  } else if (mRadii == glm::dvec3(0.0)) {
    // This will be glm::dvec3(0.0) if we failed to get values from SPICE.
    return mRadiiFromSPICE;
  }

  return mRadii;
}

void CelestialObject::setRadii(glm::dvec3 const& value) {
  mRadii = value;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

double CelestialObject::getBodyCullingRadius() const {
  return mBodyCullingRadius;
}

void CelestialObject::setBodyCullingRadius(double value) {
  mBodyCullingRadius = value;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

double CelestialObject::getOrbitCullingRadius() const {
  return mOrbitCullingRadius;
}

void CelestialObject::setOrbitCullingRadius(double value) {
  mOrbitCullingRadius = value;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool CelestialObject::getIsTrackable() const {
  return mIsTrackable;
}

void CelestialObject::setIsTrackable(bool value) {
  mIsTrackable = value;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool CelestialObject::getIsCollidable() const {
  return mIsCollidable;
}

void CelestialObject::setIsCollidable(bool value) {
  mIsCollidable = value;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void CelestialObject::setCenterName(std::string const& sCenterName) {
  CelestialAnchor::setCenterName(sCenterName);

  // We may have to get new radii from SPICE in this case.
  mRadiiFromSPICE = glm::dvec3(-1.0);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void CelestialObject::update(double tTime, cs::scene::CelestialObserver const& oObs) const {
  auto existence = getExistence();
  mIsInExistence = (tTime > existence[0] && tTime < existence[1]);

  if (getIsInExistence()) {
    try {
      matObserverRelativeTransform = oObs.getRelativeTransform(tTime, *this);
      mHasValidPosition            = true;
    } catch (...) {
      // Data might be unavailable.
      mHasValidPosition = false;
    }
  }

  mIsBodyVisible  = true;
  mIsOrbitVisible = true;

  if (mBodyCullingRadius > 0.0 || mOrbitCullingRadius > 0.0) {
    double dist = glm::length(getObserverRelativePosition());
    double size = glm::length(matObserverRelativeTransform[0]);

    if (mBodyCullingRadius > 0.0) {
      mIsBodyVisible = mBodyCullingRadius * size / dist > 0.002;
    }

    if (mOrbitCullingRadius > 0.0) {
      mIsOrbitVisible = mOrbitCullingRadius * size / dist > 0.002;
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool CelestialObject::getIsInExistence() const {
  return mIsInExistence;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool CelestialObject::getHasValidPosition() const {
  return mHasValidPosition;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool CelestialObject::getIsBodyVisible() const {
  return mIsBodyVisible && mIsInExistence && mHasValidPosition;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool CelestialObject::getIsOrbitVisible() const {
  return mIsOrbitVisible && mIsInExistence && mHasValidPosition;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

glm::dmat4 const& CelestialObject::getObserverRelativeTransform() const {
  return matObserverRelativeTransform;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

glm::dvec3 CelestialObject::getObserverRelativePosition() const {
  return matObserverRelativeTransform[3].xyz();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

glm::dmat4 CelestialObject::getObserverRelativeTransform(
    glm::dvec3 const& translation, glm::dquat const& rotation, double scale) const {

  double     angle = glm::angle(rotation);
  glm::dvec3 axis  = glm::axis(rotation);

  glm::dmat4 mat(matObserverRelativeTransform);
  mat = glm::translate(mat, translation);
  mat = glm::rotate(mat, angle, axis);
  mat = glm::scale(mat, glm::dvec3(scale, scale, scale));

  return mat;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

glm::dvec3 CelestialObject::getObserverRelativePosition(glm::dvec3 const& translation) const {
  return getObserverRelativeTransform(translation)[3].xyz();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::shared_ptr<CelestialSurface> const& CelestialObject::getSurface() const {
  return mSurface;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void CelestialObject::setSurface(std::shared_ptr<CelestialSurface> const& surface) const {
  mSurface = surface;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::shared_ptr<IntersectableObject> const& CelestialObject::getIntersectableObject() const {
  return mIntersectable;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void CelestialObject::setIntersectableObject(
    std::shared_ptr<IntersectableObject> const& object) const {
  mIntersectable = object;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cs::scene
