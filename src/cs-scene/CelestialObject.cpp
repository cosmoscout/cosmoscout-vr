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

glm::dmat4 const& CelestialObject::getWorldTransform() const {
  return matWorldTransform;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

glm::dvec4 CelestialObject::getWorldPosition() const {
  return matWorldTransform[3];
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

void CelestialObject::update(double tTime, cs::scene::CelestialObserver const& oObs) {
  mIsInExistence = (tTime > mExistence[0] && tTime < mExistence[1]);

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
