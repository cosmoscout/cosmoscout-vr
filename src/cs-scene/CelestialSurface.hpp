////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CS_SCENE_CELESTIAL_SURFACE_HPP
#define CS_SCENE_CELESTIAL_SURFACE_HPP

#include "cs_scene_export.hpp"

#include <glm/fwd.hpp>
#include <memory>

namespace cs::scene {

class CelestialObject;
class CelestialObserver;

/// A CelestialSurface can be assigned to a CelestialObject. Classes which are interested in the
/// altitude of the terrain of a celestial body can check for the existance of a CelestialSurface of
/// the respective CelestialBody. If one exists, they can call the getHeight() method in order to
/// retrieve the altitude at a given location.
class CS_SCENE_EXPORT CelestialSurface {
 public:
  /// Returns the elevation in meters at a specific point on the surface.
  ///
  /// @param lngLat The coordinates on the surface in the Geographic Coordinate System format.
  virtual double getHeight(glm::dvec2 lngLat) const = 0;
};

} // namespace cs::scene

#endif // CS_SCENE_CELESTIAL_SURFACE_HPP
