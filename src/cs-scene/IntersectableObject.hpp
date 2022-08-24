////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CS_SCENE_INTERSECTABLE_OBJECT_HPP
#define CS_SCENE_INTERSECTABLE_OBJECT_HPP

#include "cs_scene_export.hpp"

#include <glm/fwd.hpp>

namespace cs::scene {

/// An interface for objects that can be intersected by a ray. One class implementing this interface
/// is the cs::scene::CelestialObject.
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
