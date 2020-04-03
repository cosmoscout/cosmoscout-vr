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
  CelestialBody(std::string const& sCenterName, std::string const& sFrameName,
      double tStartExistence, double tEndExistence);

  CelestialBody(CelestialBody const& other) = delete;
  CelestialBody(CelestialBody&& other)      = delete;

  CelestialBody& operator=(CelestialBody const& other) = delete;
  CelestialBody& operator=(CelestialBody&& other) = delete;

  ~CelestialBody() override = default;

  /// The elevation at a specific point on the surface.
  ///
  /// @param lngLat The coordinates on the surface in the Geographic Coordinate System format.
  virtual double getHeight(glm::dvec2 lngLat) const = 0;

  /// The radii of the Body in meters.
  virtual glm::dvec3 getRadii() const = 0;
};

} // namespace cs::scene

#endif // CS_SCENE_CELESTIAL_BODY_HPP
