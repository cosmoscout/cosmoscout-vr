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

CelestialObject::CelestialObject(std::string const& sCenterName, std::string const& sFrameName,
    glm::dvec3 radii, double tStartExistence, double tEndExistence)
    : CelestialAnchor(sCenterName, sFrameName)
    , matWorldTransform(1.0)
    , mRadii(std::move(radii))
    , mStartExistence(tStartExistence)
    , mEndExistence(tEndExistence) {
}

////////////////////////////////////////////////////////////////////////////////////////////////////

glm::dmat4 const& CelestialObject::getWorldTransform() const {
  return matWorldTransform;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

glm::dvec4 CelestialObject::getWorldPosition() const {
  return matWorldTransform[3];
}

////////////////////////////////////////////////////////////////////////////////////////////////////

double CelestialObject::getStartExistence() const {
  return mStartExistence;
}

void CelestialObject::setStartExistence(double value) {
  mStartExistence = value;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

double CelestialObject::getEndExistence() const {
  return mEndExistence;
}

void CelestialObject::setEndExistence(double value) {
  mEndExistence = value;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

glm::dvec3 const& CelestialObject::getRadii() const {
  return mRadii;
}

void CelestialObject::setRadii(glm::dvec3 const& value) {
  mRadii = value;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void CelestialObject::update(double tTime, cs::scene::CelestialObserver const& oObs) {
  mIsInExistence = (tTime > mStartExistence && tTime < mEndExistence);

  if (getIsInExistence()) {
    try {
      matWorldTransform = oObs.getRelativeTransform(tTime, *this);
    } catch (...) {
      // data might be unavailable
    }
  }

  double maxRadius = glm::compMax(mRadii);
  if (maxRadius > 0) {
    double dist   = glm::length(getWorldPosition().xyz());
    double size   = maxRadius * glm::length(matWorldTransform[0]);
    double factor = size / dist;

    pVisible = factor > 0.002;
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool CelestialObject::getIsInExistence() const {
  return mIsInExistence;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cs::scene
