////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CS_UTILS_FRUSTUM_HPP
#define CS_UTILS_FRUSTUM_HPP

#include "cs_utils_export.hpp"

#include <glm/glm.hpp>

#include <array>
#include <ostream>

namespace cs::utils {

enum class FrustumPlaneIdx { eLeft = 0, eRight = 1, eBottom = 2, eTop = 3 };

std::ostream& operator<<(std::ostream& os, FrustumPlaneIdx fpi);

/// Stores a (view) frustum as the intersection of six planes. The planes' normals point inside the
/// frustum. A plane is represented as a `glm::fvec4` where the `xyz` components contain the (unit
/// length) normal and the `w` component contains the distance from the origin.
class CS_UTILS_EXPORT Frustum {
 public:
  static const size_t NUM_PLANES = 4;

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

} // namespace cs::utils

#endif // CS_UTILS_FRUSTUM_HPP
