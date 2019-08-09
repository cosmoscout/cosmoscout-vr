#include <utility>

////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "SolarSystem.hpp"

#include "../cs-utils/convert.hpp"
#include "TimeControl.hpp"

#include <VistaDataFlowNet/VdfnObjectRegistry.h>
#include <VistaKernel/Cluster/VistaClusterMode.h>
#include <VistaKernel/GraphicsManager/VistaTransformNode.h>
#include <VistaKernel/VistaSystem.h>

#include <cspice/SpiceUsr.h>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/quaternion.hpp>

namespace cs::core {

////////////////////////////////////////////////////////////////////////////////////////////////////

SolarSystem::SolarSystem(std::shared_ptr<TimeControl> const& pTimeControl)
    : mTimeControl(pTimeControl)
    , mSun(std::make_shared<scene::CelestialObject>("Sun", "IAU_Sun")) {

  pObserverCenter.onChange().connect([this](std::string const& center) {
    mObserver.changeOrigin(center, mObserver.getFrameName(), mTimeControl->pSimulationTime.get());
  });

  pObserverFrame.onChange().connect([this](std::string const& frame) {
    mObserver.changeOrigin(mObserver.getCenterName(), frame, mTimeControl->pSimulationTime.get());
  });
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::shared_ptr<const scene::CelestialObject> SolarSystem::getSun() const {
  return mSun;
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

std::shared_ptr<scene::CelestialAnchor> SolarSystem::getAnchor(std::string const& sCenter) const {
  for (auto anchor : mAnchors) {
    if (anchor->getCenterName() == sCenter) {
      return anchor;
    }
  }

  return nullptr;
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
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::set<std::shared_ptr<scene::CelestialBody>> const& SolarSystem::getBodies() const {
  return mBodies;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::shared_ptr<scene::CelestialBody> SolarSystem::getBody(std::string const& sCenter) const {
  for (auto body : mBodies) {
    if (body->getCenterName() == sCenter) {
      return body;
    }
  }

  return nullptr;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void SolarSystem::update() {
  double simulationTime(mTimeControl->pSimulationTime.get());
  double realTime(utils::convert::toSpiceTime(boost::posix_time::microsec_clock::universal_time()));
  mObserver.updateMovementAnimation(realTime);

  mSun->update(simulationTime, mObserver);
  for (auto const& object : mAnchors) {
    object->update(simulationTime, mObserver);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void SolarSystem::flyObserverTo(std::string const& sCenter, std::string const& sFrame,
    glm::dvec3 const& position, glm::dquat const& rotation, double duration) {
  // SetObserverToCamera();

  double simulationTime(mTimeControl->pSimulationTime.get());
  double startTime(
      utils::convert::toSpiceTime(boost::posix_time::microsec_clock::universal_time()));
  double endTime(startTime + duration);

  if (GetVistaSystem()->GetClusterMode()->GetIsLeader()) {
    mObserver.moveTo(sCenter, sFrame, position, rotation, simulationTime, startTime, endTime);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void SolarSystem::flyObserverTo(std::string const& sCenter, std::string const& sFrame,
    glm::dvec2 const& lngLat, double height, double duration) {
  auto radii = getRadii(sCenter);

  if (radii[0] == 0.0 || radii[2] == 0.0) {
    radii = glm::dvec3(1, 1, 1);
  }

  auto cart = utils::convert::toCartesian(lngLat, radii[0], radii[0], height);

  glm::dvec3 y = glm::dvec3(0, -1, 0);
  glm::dvec3 z = cart;
  glm::dvec3 x = glm::cross(z, y);
  y            = glm::cross(z, x);

  x = glm::normalize(x);
  y = glm::normalize(y);
  z = glm::normalize(z);

  auto rotation = glm::toQuat(glm::dmat3(x, y, z));

  flyObserverTo(sCenter, sFrame, cart, rotation, duration);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void SolarSystem::flyObserverTo(
    std::string const& sCenter, std::string const& sFrame, double duration) {
  double startTime(
      utils::convert::toSpiceTime(boost::posix_time::microsec_clock::universal_time()));
  auto radii = getRadii(sCenter);

  if (radii[0] == 0.0) {
    radii = glm::dvec3(1, 1, 1);
  }

  scene::CelestialAnchor target(sCenter, sFrame);
  auto targetDir = glm::normalize(target.getRelativePosition(startTime, mObserver));
  auto targetRot = glm::normalize(target.getRelativeRotation(startTime, mObserver));

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
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void SolarSystem::setObserverToCamera() {
  double simulationTime(mTimeControl->pSimulationTime.get());

  auto          pCam = GetVistaSystem()->GetDfnObjectRegistry()->GetObjectTransform("CAM:MAIN");
  VistaVector3D camPos;
  pCam->GetTranslation(camPos);

  scene::CelestialAnchor frame(mObserver.getCenterName(), mObserver.getFrameName());
  auto                   mat    = frame.getRelativeTransform(simulationTime, mObserver);
  glm::dvec3             offset = (mat * glm::dvec4(camPos[0], camPos[1], camPos[2], 1.0)).xyz();
  mObserver.setAnchorPosition(offset);
  pCam->SetTranslation(0, 0, 0);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void SolarSystem::printFrames() {
  SPICEINT_CELL(ids, 1000);
  bltfrm_c(SPICE_FRMTYP_ALL, &ids);

  std::cout << "-----------------------------------------" << std::endl;
  std::cout << "Built-in frames:" << std::endl;
  std::cout << "-----------------------------------------" << std::endl;

  for (int i = 0; i < card_c(&ids); ++i) {
    int obj = SPICE_CELL_ELEM_I(&ids, i);

    std::string out(50, ' ');
    frmnam_c(obj, 50, &out[0]);

    std::cout << out << std::endl;
  }

  std::cout << "-----------------------------------------" << std::endl;
  std::cout << "Loaded frames:" << std::endl;
  std::cout << "-----------------------------------------" << std::endl;

  kplfrm_c(SPICE_FRMTYP_ALL, &ids);
  for (int i = 0; i < card_c(&ids); ++i) {
    int obj = SPICE_CELL_ELEM_I(&ids, i);

    std::string out(50, ' ');
    frmnam_c(obj, 50, &out[0]);

    std::cout << out << std::endl;
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void SolarSystem::init(std::string const& sSpiceMetaFile) {
  furnsh_c(sSpiceMetaFile.c_str());
  std::string set("SET");
  std::string action("RETURN");
  erract_c(&set[0], 0, &action[0]);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void SolarSystem::cleanup() {
  kclear_c();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void SolarSystem::disableSpiceErrorReports() {
  std::string set("SET");
  std::string device("NULL");
  errdev_c(&set[0], 0, &device[0]);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void SolarSystem::enableSpiceErrorReports() {
  std::string set("SET");
  std::string device("SCREEN");
  errdev_c(&set[0], 0, &device[0]);
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

  double observerDistance =
      observer.getAnchorScale() * glm::length(observer.getRelativePosition(simulationTime, anchor));

  double scale = scaleFactor;

  if (baseDistance > 0 && observerDistance > baseDistance) {
    double diff = baseDistance * 10 - baseDistance;
    scale *= baseDistance + (1 - std::exp(-(observerDistance - baseDistance) / diff)) * diff;
  } else {
    scale *= observerDistance;
  }

  anchor.setAnchorScale(scale);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void SolarSystem::turnToObserver(scene::CelestialAnchor& anchor,
    scene::CelestialObserver const& observer, double simulationTime, bool upIsNormal) {
  // use the camera position to adjust the landmark rotation
  scene::CelestialAnchor rawAnchor(anchor.getCenterName(), anchor.getFrameName());
  rawAnchor.setAnchorPosition(anchor.getAnchorPosition());

  auto       observerTransform = rawAnchor.getRelativeTransform(simulationTime, observer);
  glm::dvec3 observerPos       = observerTransform[3];
  glm::dvec3 y                 = observerTransform * glm::dvec4(0, 1, 0, 0);
  glm::dvec3 camDir            = glm::normalize(observerPos);

  if (upIsNormal) {
    auto radii = getRadii(anchor.getCenterName());
    auto lngLat =
        cs::utils::convert::toLngLatHeight(anchor.getAnchorPosition(), radii[0], radii[0]);
    y = cs::utils::convert::lngLatToNormal(lngLat.xy(), radii[0], radii[0]);
  }

  glm::dvec3 z = glm::cross(y, camDir);
  glm::dvec3 x = glm::cross(y, z);

  x = glm::normalize(x);
  y = glm::normalize(y);
  z = glm::normalize(z);

  auto rot = glm::toQuat(glm::dmat3(x, y, z));
  anchor.setAnchorRotation(rot);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

glm::dvec3 SolarSystem::getRadii(std::string const& sCenterName) {
  // get target id code
  SpiceInt     id;
  SpiceBoolean found;
  bodn2c_c(sCenterName.c_str(), &id, &found);

  // check if radius information is available
  if (!found || !bodfnd_c(id, "RADII")) {
    return glm::dvec3(0, 0, 0);
  }

  // compute radius and convert it to meters
  SpiceInt   n;
  glm::dvec3 result;
  bodvrd_c(sCenterName.c_str(), "RADII", 3, &n, glm::value_ptr(result));
  result = result * 1000.0;

  if (n != 3) {
    throw std::runtime_error("Failed to retrieve radii for object " + sCenterName + ".");
  }

  return result;
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
