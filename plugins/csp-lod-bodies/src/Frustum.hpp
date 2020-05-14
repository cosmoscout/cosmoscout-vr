////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CSP_LOD_BODIES_FRUSTUM_HPP
#define CSP_LOD_BODIES_FRUSTUM_HPP

#include <glm/glm.hpp>

#include <array>

namespace csp::lodbodies {

enum class FrustumPlaneIdx { eLeft = 0, eRight = 1, eBottom = 2, eTop = 3, eNear = 4, eFar = 5 };

std::ostream& operator<<(std::ostream& os, FrustumPlaneIdx fpi);

/// Stores a (view) frustum as the intersection of six planes. The planes' normals point inside the
/// frustum. A plane is represented as a `glm::fvec4` where the `xyz` components contain the (unit
/// length) normal and the `w` component contains the distance from the origin.
class Frustum {
 public:
  static const size_t NUM_PLANES = 6;

  /// Constructs a new Frustum and initializes its planes from mat.
  static Frustum fromMatrix(glm::dmat4 const& mat);

  std::array<glm::dvec4, NUM_PLANES> const& getPlanes() const;

  glm::dvec4 const& getPlane(FrustumPlaneIdx fpi) const;
  void              setPlane(FrustumPlaneIdx fpi, glm::dvec4 const& plane);

  /// Sets the frustum planes from the projection matrix mat. mat may contain additional
  /// transformations (e.g. a modelview matrix) and the resulting frustum planes will be in the
  /// corresponding coordinate system.
  void setFromMatrix(glm::dmat4 const& mat);

  double getHorizontalFOV() const;
  double getVerticalFOV() const;

 private:
  std::array<glm::dvec4, NUM_PLANES> mPlanes;
};

std::ostream& operator<<(std::ostream& os, Frustum const& frustum);

} // namespace csp::lodbodies

#endif // CSP_LOD_BODIES_FRUSTUM_HPP
