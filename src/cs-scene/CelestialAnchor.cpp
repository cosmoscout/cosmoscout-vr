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
#include <optional>
#include <unordered_map>
#include <utility>

namespace cs::scene {

namespace {

////////////////////////////////////////////////////////////////////////////////////////////////////
// A little helper class which stores results of spice calls because these are quite costly. It   //
// only caches requests for one time step, as soon as the tTime parameter changes, the complete   //
// cache is invalidated.                                                                          //
////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename K, typename V>
class Cache {
 public:
  std::optional<V> get(double tTime, K const& key) {
    if (tTime != mLastTime) {
      return std::nullopt;
    }

    auto value = mache.find(key);

    if (value == mache.end()) {
      return std::nullopt;
    }

    return value->second;
  }

  void insert(double tTime, K const& key, V const& value) {
    if (tTime != mLastTime) {
      mLastTime = tTime;
      mache.clear();
    }

    mache.insert(std::make_pair(key, value));
  }

 private:
  double                   mLastTime = 0;
  std::unordered_map<K, V> mache;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace

////////////////////////////////////////////////////////////////////////////////////////////////////

CelestialAnchor::CelestialAnchor(std::string sCenterName, std::string sFrameName)
    : mPosition(0.0, 0.0, 0.0)
    , mRotation(1.0, 0.0, 0.0, 0.0)
    , mCenterName(std::move(sCenterName))
    , mFrameName(std::move(sFrameName)) {
}

////////////////////////////////////////////////////////////////////////////////////////////////////

glm::dvec3 CelestialAnchor::getRelativePosition(double tTime, CelestialAnchor const& other) const {
  glm::dvec3 vOtherPos = other.getAnchorPosition() / 1000.0;

  std::string key(getCenterName() + getFrameName() + other.getCenterName() + other.getFrameName() +
                  std::to_string(vOtherPos[0]) + std::to_string(vOtherPos[1]) +
                  std::to_string(vOtherPos[2]));

  static Cache<std::string, glm::dvec3> cache;

  auto cacheValue = cache.get(tTime, key);

  glm::dvec3 vRelPos;

  if (cacheValue) {
    vRelPos = cacheValue.value();
  } else {
    std::array<double, 6> relPos{};
    double                timeOfLight{};
    std::array            otherPos{vOtherPos[2], vOtherPos[0], vOtherPos[1]};
    spkcpt_c(otherPos.data(), other.getCenterName().c_str(), other.getFrameName().c_str(), tTime,
        mFrameName.c_str(), "OBSERVER", "NONE", mCenterName.c_str(), relPos.data(), &timeOfLight);

    if (failed_c()) {
      std::array<SpiceChar, 320> msg{};
      getmsg_c("LONG", 320, msg.data());
      reset_c();
      throw std::runtime_error(msg.data());
    }

    vRelPos = glm::dvec3(relPos[1], relPos[2], relPos[0]) * 1000.0;

    cache.insert(tTime, key, vRelPos);
  }

  vRelPos = glm::inverse(mRotation) * ((vRelPos - mPosition) / mScale);

  return vRelPos;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

glm::dquat CelestialAnchor::getRelativeRotation(double tTime, CelestialAnchor const& other) const {
  std::string key(getFrameName() + other.getFrameName());

  static Cache<std::string, glm::dquat> cache;

  auto cacheValue = cache.get(tTime, key);

  glm::dquat qRot;
  if (cacheValue) {
    qRot = cacheValue.value();
  } else {
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

    qRot = glm::angleAxis(angle, glm::dvec3(axis[1], axis[2], axis[0]));

    cache.insert(tTime, key, qRot);
  }

  return glm::inverse(mRotation) * qRot * other.mRotation;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

double CelestialAnchor::getRelativeScale(CelestialAnchor const& other) const {
  return other.mScale / mScale;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

glm::dmat4 CelestialAnchor::getRelativeTransform(double tTime, CelestialAnchor const& other) const {
  double     scale = getRelativeScale(other);
  glm::dvec3 pos   = getRelativePosition(tTime, other);
  glm::dquat rot   = getRelativeRotation(tTime, other);

  double     angle = glm::angle(rot);
  glm::dvec3 axis  = glm::axis(rot);

  glm::dmat4 mat(1.0);
  mat = glm::translate(mat, pos);
  mat = glm::rotate(mat, angle, axis);
  mat = glm::scale(mat, glm::dvec3(scale, scale, scale));

  return mat;
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

std::string const& CelestialAnchor::getCenterName() const {
  return mCenterName;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void CelestialAnchor::setCenterName(std::string const& sCenterName) {
  mCenterName = sCenterName;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

glm::dvec3 const& CelestialAnchor::getAnchorPosition() const {
  return mPosition;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void CelestialAnchor::setAnchorPosition(glm::dvec3 const& vPos) {
  mPosition = vPos;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

glm::dquat const& CelestialAnchor::getAnchorRotation() const {
  return mRotation;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void CelestialAnchor::setAnchorRotation(glm::dquat const& qRot) {
  mRotation = qRot;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

double CelestialAnchor::getAnchorScale() const {
  return mScale;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void CelestialAnchor::setAnchorScale(double dScale) {
  mScale = dScale;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void CelestialAnchor::update(double time, const CelestialObserver& observer) {
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cs::scene
