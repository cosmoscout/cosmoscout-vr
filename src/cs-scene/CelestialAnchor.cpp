////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "CelestialAnchor.hpp"

#include <VistaKernel/GraphicsManager/VistaNodeBridge.h>

#include <array>
#include <cspice/SpiceUsr.h>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/component_wise.hpp>
#include <optional>
#include <unordered_map>
#include <utility>

namespace cs::scene {

////////////////////////////////////////////////////////////////////////////////////////////////////

CelestialAnchor::CelestialAnchor(std::string sCenterName, std::string sFrameName)
    : mCenterName(std::move(sCenterName))
    , mFrameName(std::move(sFrameName)) {
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::string const& CelestialAnchor::getCenterName() const {
  return mCenterName;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void CelestialAnchor::setCenterName(std::string const& sCenterName) {
  mCenterName = sCenterName;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::string const& CelestialAnchor::getFrameName() const {
  return mFrameName;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void CelestialAnchor::setFrameName(std::string const& sFrameName) {
  mFrameName = sFrameName;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

glm::dvec3 CelestialAnchor::getRelativePosition(double tTime, CelestialAnchor const& other) const {
  std::array<double, 6> relPos{};
  double                timeOfLight{};
  std::array            otherPos{0.0, 0.0, 0.0};
  spkcpt_c(otherPos.data(), other.getCenterName().c_str(), other.getFrameName().c_str(), tTime,
      mFrameName.c_str(), "OBSERVER", "NONE", mCenterName.c_str(), relPos.data(), &timeOfLight);

  if (failed_c()) {
    std::array<SpiceChar, 320> msg{};
    getmsg_c("LONG", 320, msg.data());
    reset_c();
    throw std::runtime_error(msg.data());
  }

  return glm::dvec3(relPos[1], relPos[2], relPos[0]) * 1000.0;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

glm::dquat CelestialAnchor::getRelativeRotation(double tTime, CelestialAnchor const& other) const {

  // get rotation from self to other
  std::array<double[3], 3> rotMat{}; // NOLINT(modernize-avoid-c-arrays)
  pxform_c(other.getFrameName().c_str(), mFrameName.c_str(), tTime, rotMat.data());

  if (failed_c()) {
    std::array<SpiceChar, 320> msg{};
    getmsg_c("LONG", 320, msg.data());
    reset_c();
    throw std::runtime_error(msg.data());
  }

  // convert to quaternion
  double axis[3]; // NOLINT(modernize-avoid-c-arrays)
  double angle{};

  // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-array-to-pointer-decay, modernize-avoid-c-arrays)
  raxisa_c(rotMat.data(), axis, &angle);

  return glm::angleAxis(angle, glm::dvec3(axis[1], axis[2], axis[0]));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

glm::dmat4 CelestialAnchor::getRelativeTransform(double tTime, CelestialAnchor const& other) const {
  glm::dvec3 pos = getRelativePosition(tTime, other);
  glm::dquat rot = getRelativeRotation(tTime, other);

  double     angle = glm::angle(rot);
  glm::dvec3 axis  = glm::axis(rot);

  glm::dmat4 mat(1.0);
  mat = glm::translate(mat, pos);
  mat = glm::rotate(mat, angle, axis);

  return mat;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cs::scene
