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
  /// Creates a new CelestialBody.
  ///
  /// @param centerName  The SPICE name of the object.
  /// @param frameName   The SPICE name of the reference frame.
  /// @param radii       These will be used for visibility culling. If set to glm::dvec3(0.0),
  ///                    pVisible will not change during update().
  /// @param existence   The time range in Barycentric Dynamical Time in which the object existed.
  ///                    This should match the time coverage of the loaded SPICE kernels.
  CelestialBody(std::string const& centerName, std::string const& frameName,
      glm::dvec3 const& radii     = glm::dvec3(0.0),
      glm::dvec2 const& existence = glm::dvec2(
          std::numeric_limits<double>::lowest(), std::numeric_limits<double>::max()));

  /// Returns the elevation in meters at a specific point on the surface.
  ///
  /// @param lngLat The coordinates on the surface in the Geographic Coordinate System format.
  virtual double getHeight(glm::dvec2 lngLat) const = 0;
};

} // namespace cs::scene

#endif // CS_SCENE_CELESTIAL_BODY_HPP
