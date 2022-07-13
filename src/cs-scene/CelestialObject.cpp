////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "CelestialObject.hpp"

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

glm::dvec2 const& CelestialObject::getExistence() const {
  return mExistence;
}

void CelestialObject::setExistence(glm::dvec2 value) {
  mExistence = value;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

glm::dvec3 const& CelestialObject::getRadii() const {

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

    if (n != 3) {
      logger().warn("Failed to retrieve SPICE radii for object {}.", mCenterName);
      mRadiiFromSPICE = glm::dvec3(0.0);
    }

    // SPICE coordinates are different.
    mRadiiFromSPICE = glm::dvec3(result[1], result[2], result[0]);
    return mRadiiFromSPICE;

  } else if (mRadii == glm::dvec3(0.0)) {
    
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
  mRadiiFromSPICE = glm::dvec3(-1.0);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void CelestialObject::update(double tTime, cs::scene::CelestialObserver const& oObs) const {
  mIsInExistence = (tTime > mExistence[0] && tTime < mExistence[1]);

  if (getIsInExistence()) {
    try {
      matObserverRelativeTransform = oObs.getRelativeTransform(tTime, *this);
    } catch (...) {
      // data might be unavailable
    }
  }

  mIsBodyVisible  = true;
  mIsOrbitVisible = true;

  if (mBodyCullingRadius > 0.0 || mOrbitCullingRadius) {
    double dist = glm::length(getObserverRelativePosition().xyz());
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

bool CelestialObject::getIsBodyVisible() const {
  return mIsBodyVisible;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool CelestialObject::getIsOrbitVisible() const {
  return mIsOrbitVisible;
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

glm::dvec3 CelestialObject::getObserverRelativePosition(
    glm::dvec3 const& translation, glm::dquat const& rotation, double scale) const {
  return getObserverRelativeTransform(translation, rotation, scale)[3].xyz();
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

void CelestialObject::setIntersectableObject(std::shared_ptr<IntersectableObject> const& object) const {
  mIntersectable = object;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cs::scene
