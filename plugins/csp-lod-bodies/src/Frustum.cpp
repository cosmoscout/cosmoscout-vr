////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "Frustum.hpp"

#include "../../../src/cs-utils/utils.hpp"

#include <glm/gtx/io.hpp>

namespace csp::lodbodies {

////////////////////////////////////////////////////////////////////////////////////////////////////

std::ostream& operator<<(std::ostream& os, FrustumPlaneIdx fpi) {
  switch (fpi) {
  case FrustumPlaneIdx::eLeft:
    os << "Left";
    break;
  case FrustumPlaneIdx::eRight:
    os << "Right";
    break;
  case FrustumPlaneIdx::eBottom:
    os << "Bottom";
    break;
  case FrustumPlaneIdx::eTop:
    os << "Top";
    break;
  case FrustumPlaneIdx::eNear:
    os << "Near";
    break;
  case FrustumPlaneIdx::eFar:
    os << "Far";
    break;

    // no default - to get compiler warning when the set of enum values is
    // extended.
  };

  return os;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

/* static */ Frustum Frustum::fromMatrix(glm::dmat4 const& mat) {
  Frustum result{};
  result.setFromMatrix(mat);

  return result;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Frustum::setPlane(FrustumPlaneIdx fpi, glm::dvec4 const& plane) {
  glm::dvec3 normal(plane);
  double     len = glm::length(normal);

  mPlanes.at(cs::utils::enumCast(fpi)) =
      glm::dvec4(normal[0] / len, normal[1] / len, normal[2] / len, plane[3] / len);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Algorithm based on:
// http://fgiesen.wordpress.com/2012/08/31/frustum-planes-from-the-projection-matrix/
void Frustum::setFromMatrix(glm::dmat4 const& mat) {
  setPlane(FrustumPlaneIdx::eLeft, glm::row(mat, 3) + glm::row(mat, 0));
  setPlane(FrustumPlaneIdx::eRight, glm::row(mat, 3) - glm::row(mat, 0));

  setPlane(FrustumPlaneIdx::eBottom, glm::row(mat, 3) + glm::row(mat, 1));
  setPlane(FrustumPlaneIdx::eTop, glm::row(mat, 3) - glm::row(mat, 1));

  setPlane(FrustumPlaneIdx::eNear, glm::row(mat, 3) + glm::row(mat, 2));
  setPlane(FrustumPlaneIdx::eFar, glm::row(mat, 3) - glm::row(mat, 2));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

double Frustum::getHorizontalFOV() const {
  return glm::pi<double>() -
         std::acos(
             std::min(1.0, glm::dot(mPlanes.at(cs::utils::enumCast(FrustumPlaneIdx::eLeft)).xyz(),
                               mPlanes.at(cs::utils::enumCast(FrustumPlaneIdx::eRight)).xyz())));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

double Frustum::getVerticalFOV() const {
  return glm::pi<double>() -
         std::acos(
             std::min(1.0, glm::dot(mPlanes.at(cs::utils::enumCast(FrustumPlaneIdx::eTop)).xyz(),
                               mPlanes.at(cs::utils::enumCast(FrustumPlaneIdx::eBottom)).xyz())));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::array<glm::dvec4, Frustum::NUM_PLANES> const& Frustum::getPlanes() const {
  return mPlanes;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

glm::dvec4 const& Frustum::getPlane(FrustumPlaneIdx fpi) const {
  return mPlanes.at(cs::utils::enumCast(fpi));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::ostream& operator<<(std::ostream& os, Frustum const& frustum) {
  int index = 0;
  for (auto const& plane : frustum.getPlanes()) {
    os << ' ' << static_cast<FrustumPlaneIdx>(index++) << " (" << plane.x << ", " << plane.y << ", "
       << plane.z << ", " << plane.w << ")";
  }

  return os;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::lodbodies
