////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "CelestialObject.hpp"

#include "CelestialObserver.hpp"

namespace cs::scene {

////////////////////////////////////////////////////////////////////////////////////////////////////

CelestialObject::CelestialObject(std::string const& sCenterName, std::string const& sFrameName,
    double tStartExistence, double tEndExistence)
    : CelestialAnchor(sCenterName, sFrameName)
    , matWorldTransform(1.0)
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

void CelestialObject::update(double tTime, cs::scene::CelestialObserver const& oObs) {
  mIsInExistence = (tTime > mStartExistence && tTime < mEndExistence);

  if (getIsInExistence()) {
    try {
      matWorldTransform = oObs.getRelativeTransform(tTime, *this);
    } catch (...) {
      // data might be unavailable
    }
  }

  if (pVisibleRadius.get() > 0) {
    double dist   = glm::length(getWorldPosition().xyz());
    double size   = pVisibleRadius.get() * glm::length(matWorldTransform[0]);
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
