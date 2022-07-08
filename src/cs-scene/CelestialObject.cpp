////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "CelestialObject.hpp"

#include "CelestialObserver.hpp"

#include <glm/gtx/component_wise.hpp>

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

void CelestialObject::update(double tTime, cs::scene::CelestialObserver const& oObs) {
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

glm::dvec4 CelestialObject::getObserverRelativePosition() const {
  return matObserverRelativeTransform[3];
}

////////////////////////////////////////////////////////////////////////////////////////////////////

 std::shared_ptr<CelestialSurface> const& CelestialObject::getSurface() const {
  return mSurface;
 }

////////////////////////////////////////////////////////////////////////////////////////////////////

  void                                 CelestialObject::setSurface(std::shared_ptr<CelestialSurface> const& surface) {
    mSurface = surface;
  }

////////////////////////////////////////////////////////////////////////////////////////////////////

 std::shared_ptr<IntersectableObject> const& CelestialObject::getIntersectableObject() const {
return mIntersectable;
 }

////////////////////////////////////////////////////////////////////////////////////////////////////

  void CelestialObject::setIntersectableObject(std::shared_ptr<IntersectableObject> const& object) {
    mIntersectable = object;
  }

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cs::scene
