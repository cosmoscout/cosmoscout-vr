////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CS_SCENE_CELESTIAL_BODY_HPP
#define CS_SCENE_CELESTIAL_BODY_HPP

#include "cs_scene_export.hpp"

#include <glm/fwd.hpp>

namespace cs::scene {

/// A CelestialSurface can be assigned to a CelestialObject.
class CS_SCENE_EXPORT CelestialSurface {
 public:
  /// Returns the elevation in meters at a specific point on the surface.
  ///
  /// @param lngLat The coordinates on the surface in the Geographic Coordinate System format.
  virtual double getHeight(glm::dvec2 lngLat) const = 0;
};

} // namespace cs::scene

#endif // CS_SCENE_CELESTIAL_BODY_HPP
