#include <utility>

////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "SolarSystem.hpp"

#include "../cs-graphics/EclipseShadowMap.hpp"
#include "../cs-scene/CelestialSurface.hpp"
#include "../cs-utils/FrameTimings.hpp"
#include "../cs-utils/convert.hpp"
#include "../cs-utils/utils.hpp"
#include "GraphicsEngine.hpp"
#include "Settings.hpp"
#include "TimeControl.hpp"
#include "logger.hpp"

#include <VistaDataFlowNet/VdfnObjectRegistry.h>
#include <VistaKernel/Cluster/VistaClusterMode.h>
#include <VistaKernel/DisplayManager/VistaDisplayManager.h>
#include <VistaKernel/DisplayManager/VistaProjection.h>
#include <VistaKernel/GraphicsManager/VistaTransformNode.h>
#include <VistaKernel/VistaSystem.h>
#include <cspice/SpiceUsr.h>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/quaternion.hpp>

namespace cs::core {

////////////////////////////////////////////////////////////////////////////////////////////////////

SolarSystem::SolarSystem(std::shared_ptr<Settings> settings,
    std::shared_ptr<GraphicsEngine> graphicsEngine, std::shared_ptr<TimeControl> timeControl)
    : mSettings(std::move(settings))
    , mGraphicsEngine(std::move(graphicsEngine))
    , mTimeControl(std::move(timeControl))
    , mSun(getObject("Sun")) {

  // Tell the user what's going on.
  logger().debug("Creating SolarSystem.");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

SolarSystem::~SolarSystem() {
  // Tell the user what's going on.
  logger().debug("Deleting SolarSystem.");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::shared_ptr<const scene::CelestialObject> SolarSystem::getObject(
    std::string const& name) const {
  auto it = mSettings->mObjects.find(name);

  if (it != mSettings->mObjects.end()) {
    return it->second;
  }

  logger().error(
      "Failed to retrieve the object \"{}\": No such object is defined in the settings!", name);
  return nullptr;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::shared_ptr<const scene::CelestialObject> SolarSystem::getObjectByCenterName(
    std::string const& center) const {
  for (auto const& [name, object] : mSettings->mObjects) {
    if (object->getCenterName() == center) {
      return object;
    }
  }

  return nullptr;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::shared_ptr<const scene::CelestialObject> SolarSystem::getSun() const {
  return mSun;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

glm::dvec3 SolarSystem::getSunDirection(glm::dvec3 const& observerPosition) const {
  return glm::normalize(pSunPosition.get() - observerPosition);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

double SolarSystem::getSunIlluminance(glm::dvec3 const& observerPosition) const {
  double sunDist = glm::length(pSunPosition.get() - observerPosition);
  return pSunLuminousPower.get() / (sunDist * sunDist * 4.0 * glm::pi<double>());
}

////////////////////////////////////////////////////////////////////////////////////////////////////

double SolarSystem::getSunLuminance() const {
  double sceneScale = 1.0 / mObserver.getScale();
  double sunRadius  = mSun->getRadii()[0];

  // To get the luminous exitance (in lux) of the Sun, we have to divide its luminous power (in
  // lumens) by its surface area.
  double luminousExitance = pSunLuminousPower.get() / (sceneScale * sceneScale * sunRadius *
                                                          sunRadius * 4.0 * glm::pi<double>());

  // We consider the Sun to emit light equally in all directions. So we have to divide the
  // luminous exitance by PI to get actual luminance values.
  return luminousExitance / glm::pi<double>();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::vector<std::shared_ptr<graphics::EclipseShadowMap>> SolarSystem::getEclipseShadowMaps(
    scene::CelestialObject const& receiver, bool allowSelfShadowing) const {

  std::vector<std::shared_ptr<graphics::EclipseShadowMap>> result;

  // Loop through all registered eclipse shadow casters and test if they are casting a shadow onto
  // the given receiver. All involved objects are considered to be spheres.
  for (auto const& shadowMap : mGraphicsEngine->getEclipseShadowMaps()) {
    auto occluder = getObject(shadowMap->mOccluder);

    // Avoid self-shadowing.
    if (allowSelfShadowing || receiver.getCenterName() != occluder->getCenterName()) {

      // Get observer-centric positions.
      auto pSun = mSun->getObserverRelativePosition() * mObserver.getScale();
      auto pRec = receiver.getObserverRelativePosition() * mObserver.getScale();
      auto pOcc = occluder->getObserverRelativePosition() * mObserver.getScale();

      // Convert to receiver-centric.
      pSun = pSun - pOcc;
      pRec = pRec - pOcc;

      double dSun = glm::length(pSun);
      double dRec = glm::length(pRec);

      // Do not consider cases where the receiver is really far away.
      if (dRec > 0.1 * dSun) {
        continue;
      }

      // Do not consider cases where the receiver is in front of the caster.
      if (glm::dot(pSun / dSun, pRec / dRec) > 0) {
        continue;
      }

      double rOcc = occluder->getRadii()[0];
      double rRec = receiver.getRadii()[0];
      double rSun = mSun->getRadii()[0];

      // Compute distances to the tips of the umbra and penumbra cones.
      double dUmbra    = dSun * rOcc / (rSun - rOcc);
      double dPenumbra = dSun * rOcc / (rSun + rOcc);

      // Compute slopes of the penumbra cone.
      double mPenumbra = rOcc / std::sqrt(dPenumbra * dUmbra - rOcc * rOcc);

      // Project the vector from the occluder to the receiver onto the sun-occluder axis.
      auto toOcc        = -pRec;
      auto sunToOccNorm = -pSun / dSun;
      auto toOccProj    = glm::dot(toOcc, sunToOccNorm) * sunToOccNorm;

      // Get position in shadow space.
      double posX = glm::length(toOccProj);
      double posY = glm::length(toOcc - toOccProj);

      // Distances of the penumbra and umbra cones from the sun-occluder axis at posX.
      double penumbra = mPenumbra * (posX + dPenumbra);

      if (posY < penumbra + rRec) {
        result.push_back(shadowMap);
      }
    }
  }

  return result;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void SolarSystem::setObserver(scene::CelestialObserver const& observer) {
  mObserver = observer;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

scene::CelestialObserver& SolarSystem::getObserver() {
  return mObserver;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

scene::CelestialObserver const& SolarSystem::getObserver() const {
  return mObserver;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void SolarSystem::fixObserverFrame(double lastWorkingSimulationTime) {
  // We try getting the position of the observer relative to the origin of the Solar System. If this
  // fails, something is wrong with our observer frame.
  try {
    mObserver.getRelativePosition(mTimeControl->pSimulationTime.get(),
        scene::CelestialAnchor("Solar System Barycenter", "J2000"));
  } catch (...) {
    // In case of an error, we reset the observer. This can throw an error itself, but we cannot do
    // anything if that happens.
    try {
      mObserver.changeOrigin("Solar System Barycenter", "J2000", lastWorkingSimulationTime);
    } catch (std::exception const& e) {
      logger().error("Failed to reset the Observer SPICE frame: {}", e.what());
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void SolarSystem::update() {
  double simulationTime(mTimeControl->pSimulationTime.get());
  double realTime(
      utils::convert::time::toSpice(boost::posix_time::microsec_clock::universal_time()));
  mObserver.updateMovementAnimation(realTime);

  // First, update all celestial object positions.
  for (auto const& [name, object] : mSettings->mObjects) {
    utils::FrameTimings::ScopedTimer timer(
        "Update " + object->getCenterName() + " / " + object->getFrameName(),
        utils::FrameTimings::QueryMode::eCPU);
    object->update(simulationTime, mObserver);
  }

  // Update sun position. If a fixed Sun direction is enabled, we must calculate an artificial
  // position in the current SPICE frame at the same distance as the true Sun would be.
  auto fixedSunDist2 = glm::length2(mSettings->mGraphics.pFixedSunDirection.get());
  if (fixedSunDist2 > 0.0 && pActiveObject.get()) {
    auto trueSunDist = glm::length(mSun->getObserverRelativePosition());
    auto fixedSunDir = glm::dvec4(mSettings->mGraphics.pFixedSunDirection.get(), 0.0);
    pSunPosition =
        glm::normalize((pActiveObject()->getObserverRelativeTransform() * fixedSunDir).xyz()) *
        trueSunDist;
  } else {
    pSunPosition = mSun->getObserverRelativePosition();
  }

  // Calculate luminous power of the Sun. This can be calculated by multiplying the illuminance at
  // the average distance of Earth with the surface area of a sphere with a radius of the average
  // distance of Earth.

  // Luminous power of the Sun in lumens. Number is taken from
  // Darula, Stan, Richard Kittler, and Christian A. Gueymard. "Reference luminous solar constant
  // and solar luminance for illuminance calculations." Solar Energy 79.5 (2005)
  double const sunLuminousPower = 3.75e28;

  // As our scene is always scaled, we have to scale the luminous power of the sun accordingly.
  // Else, our Sun would be extremely bright when scaled down.
  double sceneScale = 1.0 / mObserver.getScale();
  pSunLuminousPower = static_cast<float>(sunLuminousPower * sceneScale * sceneScale);

  // Update the property containing the current observer speed.
  auto observerPosition = mObserver.getPosition();
  auto now              = std::chrono::high_resolution_clock::now();

  auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(now - mLastTime).count();

  // Duration is in nanoseconds so we have to multiply by 1.0e9.
  if (duration > 0) {
    if (!mSpiceFrameChangedLastFrame) {
      double const secToNano = 1.0e9;
      pCurrentObserverSpeed =
          static_cast<float>(secToNano * glm::length(mLastPosition - observerPosition) / duration);
    }
    mLastPosition = observerPosition;
    mLastTime     = now;
  }

  // Update settings properties.
  mSettings->mObserver.pCenter   = mObserver.getCenterName();
  mSettings->mObserver.pFrame    = mObserver.getFrameName();
  mSettings->mObserver.pPosition = mObserver.getPosition();
  mSettings->mObserver.pRotation = mObserver.getRotation();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void SolarSystem::updateSceneScale() {

  // First we have to find the planet which is closest to the observer.
  std::shared_ptr<const scene::CelestialObject> closestObject;
  double dClosestDistance = std::numeric_limits<double>::max();

  // Here we will store the position of the observer relative to the closestObject.
  glm::dvec3 vClosestPlanetObserverPosition(0.0);

  for (auto const& [name, object] : mSettings->mObjects) {

    // Skip non-existent objects.
    if (!object->getIsInExistence() || !object->getHasValidPosition() ||
        !object->getIsTrackable()) {
      continue;
    }

    // Skip objects with an unknown radius.
    auto radii = object->getRadii() * object->getScale();
    if (radii.x <= 0.0 || radii.y <= 0.0 || radii.z <= 0.0) {
      continue;
    }

    // Finally check if the current body is closest to the observer. We won't incorporate surface
    // elevation in this check.
    auto vObserverPos =
        glm::inverse(object->getObserverRelativeTransform()) * glm::dvec4(0.0, 0.0, 0.0, 1.0);
    double dDistance = glm::length(vObserverPos) - radii[0];

    if (dDistance < dClosestDistance) {
      closestObject                  = object;
      dClosestDistance               = dDistance;
      vClosestPlanetObserverPosition = vObserverPos;
    }
  }

  // Now that we found a closest body, we will scale the observer in such a way, that the closest
  // body is rendered at a distance between mSettings->mSceneScale.mCloseVisualDistance and
  // mSettings->mSceneScale.mFarVisualDistance (in meters).
  if (closestObject) {

    // First we calculate the *real* world-space distance to the planet (incorporating surface
    // elevation).
    auto radii = closestObject->getRadii() * closestObject->getScale();
    auto lngLatHeight =
        cs::utils::convert::cartesianToLngLatHeight(vClosestPlanetObserverPosition, radii);
    double dRealDistance = lngLatHeight.z;

    if (closestObject->getSurface()) {
      dRealDistance -= closestObject->getSurface()->getHeight(lngLatHeight.xy()) *
                       mSettings->mGraphics.pHeightScale.get();
    }

    if (std::isnan(dRealDistance)) {
      return;
    }

    // The render distance between mSettings->mSceneScale.mCloseVisualDistance and
    // mSettings->mSceneScale.mFarVisualDistance is chosen based on the observer's world-space
    // distance between mSettings->mSceneScale.mFarRealDistance and
    // mSettings->mSceneScale.mCloseRealDistance (also in meters).
    double interpolate = 1.0;

    if (mSettings->mSceneScale.mFarRealDistance != mSettings->mSceneScale.mCloseRealDistance) {
      interpolate = glm::clamp(
          (dRealDistance - mSettings->mSceneScale.mCloseRealDistance) /
              (mSettings->mSceneScale.mFarRealDistance - mSettings->mSceneScale.mCloseRealDistance),
          0.0, 1.0);
    }

    double dScale = dRealDistance / glm::mix(mSettings->mSceneScale.mCloseVisualDistance,
                                        mSettings->mSceneScale.mFarVisualDistance, interpolate);
    dScale = glm::clamp(dScale, mSettings->mSceneScale.mMinScale, mSettings->mSceneScale.mMaxScale);
    mObserver.setScale(dScale);

    if (dRealDistance < mSettings->mSceneScale.mCloseRealDistance &&
        closestObject->getIsCollidable()) {
      double     penetration = mSettings->mSceneScale.mCloseRealDistance - dRealDistance;
      glm::dvec3 position    = mObserver.getPosition();
      mObserver.setPosition(position + glm::normalize(position) * penetration);
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void SolarSystem::updateObserverFrame() {

  // The Observer will be locked to the active planet.
  std::shared_ptr<const scene::CelestialObject> activeObject;

  // The active planet is the one with the highest *weight*.
  double dActiveWeight = 0;

  for (auto const& [name, object] : mSettings->mObjects) {
    // Skip non-existant objects.
    if (!object->getIsInExistence() || !object->getHasValidPosition() ||
        !object->getIsTrackable()) {
      continue;
    }

    // Skip objects with an unknown radius.
    auto radii = object->getRadii() * object->getScale();
    if (radii.x <= 0.0 || radii.y <= 0.0 || radii.z <= 0.0) {
      continue;
    }

    double dDistance =
        glm::length(object->getObserverRelativePosition() * mObserver.getScale()) - radii[0];

    // The weight depends on the object size and its distance to the observer.
    double dWeight = (radii[0] + mSettings->mSceneScale.mMinObjectSize) /
                     std::max(radii[0] + mSettings->mSceneScale.mMinObjectSize,
                         radii[0] + dDistance - mSettings->mSceneScale.mMinObjectSize);

    // The Sun is quite huge. We reduce its weight a bit so that the observer is more inclined to
    // stay at planets.
    if (object == mSun) {
      dWeight *= 0.01;
    }

    if (dWeight > dActiveWeight && (dWeight > mSettings->mSceneScale.mLockWeight ||
                                       dWeight > mSettings->mSceneScale.mTrackWeight)) {
      activeObject  = object;
      dActiveWeight = dWeight;
    }
  }

  // If currently no observer animation is in progress, we change the pActiveObject accordingly.
  // This may be null if we are very far away from any object.
  if (!mObserver.isAnimationInProgress()) {
    pActiveObject = activeObject;

    std::string sCenter = "Solar System Barycenter";
    std::string sFrame  = "J2000";

    // We change frame and center if there is an object with weight larger than mLockWeight
    // and mTrackWeight.
    if (activeObject) {
      if (dActiveWeight > mSettings->mSceneScale.mLockWeight) {
        sFrame = activeObject->getFrameName();
      }

      if (dActiveWeight > mSettings->mSceneScale.mTrackWeight) {
        sCenter = activeObject->getCenterName();
      }
    }

    if (sCenter != mObserver.getCenterName() || sFrame != mObserver.getFrameName()) {
      mObserver.changeOrigin(sCenter, sFrame, mTimeControl->pSimulationTime.get());
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void SolarSystem::flyObserverTo(std::string const& sCenter, std::string const& sFrame,
    glm::dvec3 const& position, glm::dquat const& rotation, double duration) {

  double simulationTime(mTimeControl->pSimulationTime.get());
  double startTime(
      utils::convert::time::toSpice(boost::posix_time::microsec_clock::universal_time()));
  double endTime(startTime + duration);

  if (GetVistaSystem()->GetClusterMode()->GetIsLeader()) {
    mObserver.moveTo(sCenter, sFrame, position, rotation, simulationTime, startTime, endTime);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void SolarSystem::flyObserverTo(std::string const& sCenter, std::string const& sFrame,
    glm::dvec3 const& position, double duration) {

  glm::dvec3 y = glm::dvec3(0, -1, 0);
  glm::dvec3 z = position;
  glm::dvec3 x = glm::cross(z, y);
  y            = glm::cross(z, x);

  x = glm::normalize(x);
  y = glm::normalize(y);
  z = glm::normalize(z);

  auto rotation = glm::toQuat(glm::dmat3(x, y, z));

  flyObserverTo(sCenter, sFrame, position, rotation, duration);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void SolarSystem::flyObserverTo(std::string const& sCenter, std::string const& sFrame,
    glm::dvec2 const& lngLat, double height, double duration) {

  auto       object = getObjectByCenterName(sCenter);
  glm::dvec3 radii(1.0);

  if (object) {
    auto r = object->getRadii();

    if (radii[0] > 0.0 && radii[1] > 0.0 && radii[2] > 0.0) {
      radii = r;
    }
  }

  auto cart = utils::convert::toCartesian(lngLat, radii, height);

  flyObserverTo(sCenter, sFrame, cart, duration);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void SolarSystem::flyObserverTo(
    std::string const& sCenter, std::string const& sFrame, double duration) {

  try {
    auto       object = getObjectByCenterName(sCenter);
    glm::dvec3 radii(1.0);

    if (object) {
      auto r = object->getRadii();

      if (radii[0] > 0.0 && radii[1] > 0.0 && radii[2] > 0.0) {
        radii = r;
      }
    }

    scene::CelestialAnchor target(sCenter, sFrame);

    auto targetDir =
        glm::normalize(target.getRelativePosition(mTimeControl->pSimulationTime.get(), mObserver));
    auto targetRot =
        glm::normalize(target.getRelativeRotation(mTimeControl->pSimulationTime.get(), mObserver));

    auto cart = targetDir * radii[0] * 3.0;

    glm::dvec3 y = targetRot * glm::dvec3(0, -1, 0);
    glm::dvec3 z = cart;
    glm::dvec3 x = glm::cross(z, y);
    y            = glm::cross(z, x);

    x = glm::normalize(x);
    y = glm::normalize(y);
    z = glm::normalize(z);

    auto rotation = glm::toQuat(glm::dmat3(x, y, z));

    flyObserverTo(sCenter, sFrame, cart, rotation, duration);

  } catch (std::exception const& e) {
    // Getting the relative transformation may fail due to insufficient SPICE data.
    logger().warn("SolarSystem::flyObserverTo failed: {}", e.what());
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void SolarSystem::printFrames() {
  SPICEINT_CELL(ids, 1000); // NOLINT: Creates a c-array.
  bltfrm_c(SPICE_FRMTYP_ALL, &ids);

  logger().info("-----------------------------------------");
  logger().info("Built-in frames:");
  logger().info("-----------------------------------------");

  int64_t const length = 50;

  for (int i = 0; i < card_c(&ids); ++i) {
    int         obj = SPICE_CELL_ELEM_I(&ids, i); // NOLINT
    std::string out(length, ' ');
    frmnam_c(obj, length, out.data());

    logger().info(out);
  }

  logger().info("-----------------------------------------");
  logger().info("Loaded frames:");
  logger().info("-----------------------------------------");

  kplfrm_c(SPICE_FRMTYP_ALL, &ids); // NOLINT
  for (int i = 0; i < card_c(&ids); ++i) {
    int obj = SPICE_CELL_ELEM_I(&ids, i); // NOLINT

    std::string out(length, ' ');
    frmnam_c(obj, length, out.data());

    logger().info(out);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void SolarSystem::init(std::string const& sSpiceMetaFile) {

  std::string actionReturn = "RETURN";
  // Continue execution on errors.
  erract_c("SET", 0, actionReturn.data());

  std::string actionNull = "NULL";
  // Disable default error reports.
  errdev_c("SET", 0, actionNull.data());

  // Load the spice kernels.
  furnsh_c(sSpiceMetaFile.c_str());

  if (failed_c()) {
    int32_t const maxSpiceErrorLength = 320;

    std::array<SpiceChar, maxSpiceErrorLength> msg{};
    getmsg_c("LONG", maxSpiceErrorLength, msg.data());
    throw std::runtime_error(msg.data());
  }

  mIsInitialized = true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool SolarSystem::getIsInitialized() const {
  return mIsInitialized;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void SolarSystem::deinit() {
  kclear_c();
  mIsInitialized = false;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

double SolarSystem::getScaleBasedOnObserverDistance(
    std::shared_ptr<const scene::CelestialObject> const& object, glm::dvec3 const& translation,
    double baseDistance, double scaleFactor) {

  double observerDistance =
      mObserver.getScale() * glm::length(object->getObserverRelativePosition(translation));

  double scale = scaleFactor;

  if (baseDistance > 0 && observerDistance > baseDistance) {
    double diff = baseDistance * 10 - baseDistance;
    scale *= baseDistance + (1 - std::exp(-(observerDistance - baseDistance) / diff)) * diff;
  } else {
    scale *= observerDistance;
  }

  return scale;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

glm::dquat SolarSystem::getRotationToObserver(
    std::shared_ptr<const scene::CelestialObject> const& object, glm::dvec3 const& translation,
    bool upIsNormal) {

  auto       observerTransform = glm::inverse(object->getObserverRelativeTransform(translation));
  glm::dvec3 observerPos       = observerTransform[3];
  glm::dvec3 y                 = observerTransform * glm::dvec4(0, 1, 0, 0);
  glm::dvec3 camDir            = glm::normalize(observerPos);

  if (upIsNormal) {
    auto radii  = object->getRadii();
    auto lngLat = cs::utils::convert::cartesianToLngLat(translation, radii);
    y           = cs::utils::convert::lngLatToNormal(lngLat);
  }

  glm::dvec3 z = glm::cross(y, camDir);
  glm::dvec3 x = glm::cross(y, z);

  x = glm::normalize(x);
  y = glm::normalize(y);
  z = glm::normalize(z);

  return glm::toQuat(glm::dmat3(x, y, z));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::vector<glm::dvec4> SolarSystem::calculateTrajectory(std::string const& sCenterName,
    std::string const& sFrameName, std::string const& sTargetName, double dStartTime,
    double dEndTime, int iSamples) {
  std::vector<glm::dvec4> vPoints;

  double dLength(dEndTime - dStartTime);

  scene::CelestialAnchor obs(sCenterName, sFrameName);
  scene::CelestialAnchor body(sTargetName, sFrameName);

  for (int i(0); i < iSamples; ++i) {
    double dTime = dLength / iSamples * i;
    try {
      glm::dvec3 pos = obs.getRelativePosition(dTime + dStartTime, body);
      vPoints.emplace_back(glm::dvec4(pos.x, pos.y, pos.z, dTime + dStartTime));
    } catch (...) {}
  }

  return vPoints;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cs::core
