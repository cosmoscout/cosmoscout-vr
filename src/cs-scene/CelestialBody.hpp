////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CS_SCENE_CELESTIAL_BODY_HPP
#define CS_SCENE_CELESTIAL_BODY_HPP

#include "../cs-utils/IntersectableObject.hpp"
#include "CelestialObject.hpp"

namespace cs::scene {

/// CelestialBody objects extend the CelestialObject by being intersectable. Every implementation
/// needs to override the getIntersection() method from utils::IntersectableObject.
class CS_SCENE_EXPORT CelestialBody : public CelestialObject, public utils::IntersectableObject {
 public:
  /// If set to false, the SolarSystem will not consider this body for the computation of the active
  /// body.
  utils::Property<bool> pTrackable = true;

  /// Returns the elevation in meters at a specific point on the surface.
  ///
  /// @param lngLat The coordinates on the surface in the Geographic Coordinate System format.
  virtual double getHeight(glm::dvec2 lngLat) const = 0;
};

} // namespace cs::scene

#endif // CS_SCENE_CELESTIAL_BODY_HPP
