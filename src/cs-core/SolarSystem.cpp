#include <utility>

////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "SolarSystem.hpp"

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
    std::shared_ptr<utils::FrameTimings>           frameTimings,
    std::shared_ptr<GraphicsEngine> graphicsEngine, std::shared_ptr<TimeControl> timeControl)
    : mSettings(std::move(settings))
    , mFrameTimings(std::move(frameTimings))
    , mGraphicsEngine(std::move(graphicsEngine))
    , mTimeControl(std::move(timeControl))
    , mSun(std::make_shared<scene::CelestialObject>()) {

  mSun->setCenterName("Sun");
  mSun->setFrameName("IAU_Sun");

  // Tell the user what's going on.
  logger().debug("Creating SolarSystem.");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

SolarSystem::~SolarSystem() {
  // Tell the user what's going on.
  logger().debug("Deleting SolarSystem.");
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

void SolarSystem::registerAnchor(std::shared_ptr<scene::CelestialAnchor> const& anchor) {
  mAnchors.insert(anchor);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void SolarSystem::unregisterAnchor(std::shared_ptr<scene::CelestialAnchor> const& anchor) {
  mAnchors.erase(anchor);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::set<std::shared_ptr<scene::CelestialAnchor>> const& SolarSystem::getAnchors() const {
  return mAnchors;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void SolarSystem::registerBody(std::shared_ptr<scene::CelestialBody> const& body) {
  mBodies.insert(body);
  mAnchors.insert(body);

  for (const auto& listener : mAddBodyListeners) {
    listener.second(body);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void SolarSystem::unregisterBody(std::shared_ptr<scene::CelestialBody> const& body) {
  mBodies.erase(body);
  mAnchors.erase(body);

  for (const auto& listener : mRemoveBodyListeners) {
    listener.second(body);
  }

  if (pActiveBody.get() == body) {
    pActiveBody = nullptr;
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::set<std::shared_ptr<scene::CelestialBody>> const& SolarSystem::getBodies() const {
  return mBodies;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::shared_ptr<scene::CelestialBody> SolarSystem::getBody(std::string sCenter) const {
  std::transform(sCenter.begin(), sCenter.end(), sCenter.begin(),
      [](unsigned char c) { return std::tolower(c); });

  for (auto body : mBodies) {
    auto name = body->getCenterName();
    std::transform(
        name.begin(), name.end(), name.begin(), [](unsigned char c) { return std::tolower(c); });

    if (name == sCenter) {
      return body;
    }
  }

  return nullptr;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void SolarSystem::update() {
  double simulationTime(mTimeControl->pSimulationTime.get());
  double realTime(
      utils::convert::time::toSpice(boost::posix_time::microsec_clock::universal_time()));
  mObserver.updateMovementAnimation(realTime);

  mSun->update(simulationTime, mObserver);

  for (auto const& object : mAnchors) {
    object->update(simulationTime, mObserver);
  }

  // Update sun position. If a fixed Sun direction is enabled, we must calculate an artificial
  // position in the current SPICE frame at the same distance as the true Sun would be.
  auto fixedSunDist2 = glm::length2(mSettings->mGraphics.pFixedSunDirection.get());
  if (fixedSunDist2 > 0.0 && pActiveBody.get()) {
    auto trueSunDist = glm::length(mSun->getWorldTransform()[3].xyz());
    auto fixedSunDir = glm::dvec4(mSettings->mGraphics.pFixedSunDirection.get(), 0.0);
    pSunPosition =
        glm::normalize((pActiveBody()->getWorldTransform() * fixedSunDir).xyz()) * trueSunDist;
  } else {
    pSunPosition = mSun->getWorldTransform()[3].xyz();
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
  double sceneScale = 1.0 / mObserver.getAnchorScale();
  pSunLuminousPower = static_cast<float>(sunLuminousPower * sceneScale * sceneScale);

  // Update the property containing the current observer speed.
  auto observerPosition = mObserver.getAnchorPosition();
  auto now              = std::chrono::high_resolution_clock::now();

  auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(now - mLastTime).count();

  // Duration is in nanoseconds so we have to multiply by 1.0e9.
  if (duration > 0) {
    double const secToNano = 1.0e9;
    pCurrentObserverSpeed =
        static_cast<float>(secToNano * glm::length(mLastPosition - observerPosition) / duration);
    mLastPosition = observerPosition;
    mLastTime     = now;
  }

  // Update settings properties.
  mSettings->mObserver.pCenter   = mObserver.getCenterName();
  mSettings->mObserver.pFrame    = mObserver.getFrameName();
  mSettings->mObserver.pPosition = mObserver.getAnchorPosition();
  mSettings->mObserver.pRotation = mObserver.getAnchorRotation();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void SolarSystem::updateSceneScale() {

  // First we have to find the planet which is closest to the observer.
  std::shared_ptr<cs::scene::CelestialBody> closestBody;
  double                                    dClosestDistance = std::numeric_limits<double>::max();

  // Here we will store the position of the observer relative to the closestBody.
  glm::dvec3 vClosestPlanetObserverPosition(0.0);

  for (auto const& object : getBodies()) {

    // Skip non-existant objects.
    if (!object->getIsInExistence() || !object->pTrackable.get()) {
      continue;
    }

    // Skip objects with an unkown radius.
    auto radii = object->getRadii() * object->getAnchorScale();
    if (radii.x <= 0.0 || radii.y <= 0.0 || radii.z <= 0.0) {
      continue;
    }

    // Finally check if the current body is closest to the observer. We won't incorporate surface
    // elevation in this check.
    try {
      auto vObserverPos =
          object->getRelativePosition(mTimeControl->pSimulationTime.get(), mObserver) *
          object->getAnchorScale();
      double dDistance = glm::length(vObserverPos) - radii[0];

      if (dDistance < dClosestDistance) {
        closestBody                    = object;
        dClosestDistance               = dDistance;
        vClosestPlanetObserverPosition = vObserverPos;
      }
    } catch (...) {
      // If getting the relative position of the body failed, this may be due to two reasons: Either
      // we do not have suffcient SPICE data for the SPICE frame of the body or we do not have
      // enough data for the observer frame. The former issue is ok, we will just skip this body and
      // try the next one. The latter is more tricky as this will cause issues with all bodies and
      // we won't find any object to scale the scene to. However, the Application will catch an
      // error from the SolarSystem::update() call and will call SolarSystem::fixObserverFrame() to
      // get the observer back to a valid frame.
      continue;
    }
  }

  // Now that we found a closest body, we will scale the observer in such a way, that the closest
  // body is rendered at a distance between mSettings->mSceneScale.mCloseVisualDistance and
  // mSettings->mSceneScale.mFarVisualDistance (in meters).
  if (closestBody) {

    // First we calculate the *real* world-space distance to the planet (incorporating surface
    // elevation).
    auto radii = closestBody->getRadii() * closestBody->getAnchorScale();
    auto lngLatHeight =
        cs::utils::convert::cartesianToLngLatHeight(vClosestPlanetObserverPosition, radii);
    double dRealDistance = lngLatHeight.z - closestBody->getHeight(lngLatHeight.xy()) *
                                                mSettings->mGraphics.pHeightScale.get();

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
    mObserver.setAnchorScale(dScale);

    if (dRealDistance < mSettings->mSceneScale.mCloseRealDistance) {
      double     penetration = mSettings->mSceneScale.mCloseRealDistance - dRealDistance;
      glm::dvec3 position    = mObserver.getAnchorPosition();
      mObserver.setAnchorPosition(position + glm::normalize(position) * penetration);
    }

    // We set the far clip plane dynamically, based on the same interpolation factor.
    auto projections = GetVistaSystem()->GetDisplayManager()->GetProjections();
    for (auto const& projection : projections) {
      projection.second->GetProjectionProperties()->SetClippingRange(
          mSettings->mSceneScale.mNearClip, glm::mix(mSettings->mSceneScale.mMaxFarClip,
                                                mSettings->mSceneScale.mMinFarClip, interpolate));
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void SolarSystem::updateObserverFrame() {

  // The Observer will be locked to the active planet.
  std::shared_ptr<cs::scene::CelestialBody> activeBody;

  // The active planet is the one with the heighest *weight*.
  double dActiveWeight = 0;

  for (auto const& object : getBodies()) {
    // Skip non-existant objects.
    if (!object->getIsInExistence() || !object->pTrackable.get()) {
      continue;
    }

    // Skip objects with an unkown radius.
    auto radii = object->getRadii() * object->getAnchorScale();
    if (radii.x <= 0.0 || radii.y <= 0.0 || radii.z <= 0.0) {
      continue;
    }

    try {
      auto vObserverPos =
          object->getRelativePosition(mTimeControl->pSimulationTime.get(), mObserver) *
          object->getAnchorScale();
      double dDistance = glm::length(vObserverPos) - radii[0];

      // The weigh depends on the object size and it's distance to the observer.
      double dWeight = (radii[0] + mSettings->mSceneScale.mMinObjectSize) /
                       std::max(radii[0] + mSettings->mSceneScale.mMinObjectSize,
                           radii[0] + dDistance - mSettings->mSceneScale.mMinObjectSize);

      // The Sun is quite huge. We reduce it's weight a bit so that the observer is more inclined to
      // stay at planets.
      if (object->getCenterName() == "Sun") {
        dWeight *= 0.01;
      }

      if (dWeight > dActiveWeight) {
        activeBody    = object;
        dActiveWeight = dWeight;
      }
    } catch (...) {
      // If getting the relative position of the body failed, this may be due to two reasons: Either
      // we do not have suffcient SPICE data for the SPICE frame of the body or we do not have
      // enough data for the observer frame. The former issue is ok, we will just skip this body and
      // try the next one. The latter is more tricky as this will cause issues with all bodies and
      // we won't find any active object to track. However, the Application will catch an error from
      // the SolarSystem::update() call and will call SolarSystem::fixObserverFrame() to get the
      // observer back to a valid frame.
      continue;
    }
  }

  // We change frame and center if there is a object with weight larger than mLockWeight
  // and mTrackWeight.
  if (activeBody) {
    if (!mObserver.isAnimationInProgress()) {
      std::string sCenter = "Solar System Barycenter";
      std::string sFrame  = "J2000";

      if (dActiveWeight > mSettings->mSceneScale.mLockWeight) {
        sFrame = activeBody->getFrameName();
      }

      if (dActiveWeight > mSettings->mSceneScale.mTrackWeight) {
        sCenter = activeBody->getCenterName();
      }

      pActiveBody = activeBody;

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
  auto radii = getRadii(sCenter);

  if (radii[0] == 0.0 || radii[2] == 0.0) {
    radii = glm::dvec3(1, 1, 1);
  }

  auto cart = utils::convert::toCartesian(lngLat, radii, height);

  flyObserverTo(sCenter, sFrame, cart, duration);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void SolarSystem::flyObserverTo(
    std::string const& sCenter, std::string const& sFrame, double duration) {

  try {
    auto radii = getRadii(sCenter);

    if (radii[0] == 0.0) {
      radii = glm::dvec3(1, 1, 1);
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

uint64_t SolarSystem::registerAddBodyListener(
    std::function<void(std::shared_ptr<scene::CelestialBody>)> listener) {
  auto id               = mListenerIds++;
  mAddBodyListeners[id] = std::move(listener);
  return id;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void SolarSystem::unregisterAddBodyListener(uint64_t id) {
  mAddBodyListeners.erase(id);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

uint64_t SolarSystem::registerRemoveBodyListener(
    std::function<void(std::shared_ptr<scene::CelestialBody>)> listener) {
  auto id                  = mListenerIds++;
  mRemoveBodyListeners[id] = std::move(listener);
  return id;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void SolarSystem::unregisterRemoveBodyListener(uint64_t id) {
  mRemoveBodyListeners.erase(id);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void SolarSystem::scaleRelativeToObserver(scene::CelestialAnchor& anchor,
    scene::CelestialObserver const& observer, double simulationTime, double baseDistance,
    double scaleFactor) {

  try {
    double observerDistance = observer.getAnchorScale() *
                              glm::length(observer.getRelativePosition(simulationTime, anchor));

    double scale = scaleFactor;

    if (baseDistance > 0 && observerDistance > baseDistance) {
      double diff = baseDistance * 10 - baseDistance;
      scale *= baseDistance + (1 - std::exp(-(observerDistance - baseDistance) / diff)) * diff;
    } else {
      scale *= observerDistance;
    }

    anchor.setAnchorScale(scale);
  } catch (std::exception const& e) {
    // Getting the relative transformation may fail due to insufficient SPICE data.
    logger().warn("SolarSystem::scaleRelativeToObserver failed: {}", e.what());
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void SolarSystem::turnToObserver(scene::CelestialAnchor& anchor,
    scene::CelestialObserver const& observer, double simulationTime, bool upIsNormal) {
  // use the camera position to adjust the landmark rotation
  scene::CelestialAnchor rawAnchor(anchor.getCenterName(), anchor.getFrameName());
  rawAnchor.setAnchorPosition(anchor.getAnchorPosition());

  try {
    auto       observerTransform = rawAnchor.getRelativeTransform(simulationTime, observer);
    glm::dvec3 observerPos       = observerTransform[3];
    glm::dvec3 y                 = observerTransform * glm::dvec4(0, 1, 0, 0);
    glm::dvec3 camDir            = glm::normalize(observerPos);

    if (upIsNormal) {
      auto radii  = getRadii(anchor.getCenterName());
      auto lngLat = cs::utils::convert::cartesianToLngLat(anchor.getAnchorPosition(), radii);
      y           = cs::utils::convert::lngLatToNormal(lngLat);
    }

    glm::dvec3 z = glm::cross(y, camDir);
    glm::dvec3 x = glm::cross(y, z);

    x = glm::normalize(x);
    y = glm::normalize(y);
    z = glm::normalize(z);

    auto rot = glm::toQuat(glm::dmat3(x, y, z));
    anchor.setAnchorRotation(rot);

  } catch (std::exception const& e) {
    // Getting the relative transformation may fail due to insufficient SPICE data.
    logger().warn("SolarSystem::turnToObserver failed: {}", e.what());
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

glm::dvec3 SolarSystem::getRadii(std::string const& sCenterName) {
  // get target id code
  SpiceInt     id{};
  SpiceBoolean found{};
  bodn2c_c(sCenterName.c_str(), &id, &found);

  // check if radius information is available
  if (!found || !bodfnd_c(id, "RADII")) {
    return glm::dvec3(0, 0, 0);
  }

  // compute radius and convert it to meters
  SpiceInt   n{};
  glm::dvec3 result;
  bodvrd_c(sCenterName.c_str(), "RADII", 3, &n, glm::value_ptr(result));
  double const kmToMeter = 1000.0;
  result                 = result * kmToMeter;

  if (n != 3) {
    throw std::runtime_error("Failed to retrieve SPICE radii for object " + sCenterName + ".");
  }

  // SPICE coordinates are different.
  return glm::dvec3(result[1], result[2], result[0]);
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
