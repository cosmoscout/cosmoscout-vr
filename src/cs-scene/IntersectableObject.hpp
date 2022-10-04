////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CS_SCENE_INTERSECTABLE_OBJECT_HPP
#define CS_SCENE_INTERSECTABLE_OBJECT_HPP

#include "cs_scene_export.hpp"

#include <glm/fwd.hpp>

namespace cs::scene {

/// An interface for objects that can be intersected by a ray. A class deriving from
/// IntersectableObject can be assigned to a CelestialObject. Once each frame, the InputManager will
/// check all CelestialObjects for if they contain an IntersectableObject. If so, it will be tested
/// for intersections with the mouse ray.
class IntersectableObject {
 public:
  /// Calculates the intersection of the implementing object and a ray.
  /// @param      rayPos  The origin of the ray.
  /// @param      rayDir  The direction of the ray.
  /// @param[out] pos     The point of intersection if one exists.
  /// @return             If an intersection happened.
  virtual bool getIntersection(
      glm::dvec3 const& rayPos, glm::dvec3 const& rayDir, glm::dvec3& pos) const = 0;
};

} // namespace cs::scene

#endif // CS_SCENE_INTERSECTABLE_OBJECT_HPP
