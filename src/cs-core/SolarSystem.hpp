////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CS_CORE_SOLARSYSTEM_HPP
#define CS_CORE_SOLARSYSTEM_HPP

#include "cs_core_export.hpp"

#include "../cs-scene/CelestialAnchor.hpp"
#include "../cs-scene/CelestialBody.hpp"
#include "../cs-scene/CelestialObserver.hpp"

#include <chrono>
#include <set>
#include <unordered_set>
#include <vector>

namespace cs::utils {
class FrameTimings;
} // namespace cs::utils

namespace cs::core {

class Settings;
class TimeControl;
class GraphicsEngine;

/// The solar system is responsible for managing all CelestialBodies, the Sun and the observer.
/// It functions as an interface to access these above mentioned objects from nearly anywhere in
/// the application.
class CS_CORE_EXPORT SolarSystem {
 public:
  /// The body which the observer is attached to. The observer will follow this bodies motions.
  utils::Property<std::shared_ptr<scene::CelestialBody>> pActiveBody;

  /// The name of the current SPICE center of the observer. Should be the same as
  /// pActiveBody.get()->getCenterName()
  utils::Property<std::string> pObserverCenter;

  /// The SPICE frame of reference the observer is currently in.
  utils::Property<std::string> pObserverFrame;

  /// The current speed of the observer in m/s in relation to his current SPICE reference frame.
  utils::Property<float> pCurrentObserverSpeed;

  SolarSystem(std::shared_ptr<const Settings> const& settings,
      std::shared_ptr<utils::FrameTimings> const&    frameTimings,
      std::shared_ptr<GraphicsEngine> const&         graphicsEngine,
      std::shared_ptr<TimeControl> const&            timeControl);
  ~SolarSystem();

  /// The Sun which is at the center of the SolarSystem.
  std::shared_ptr<const scene::CelestialObject> getSun() const;

  /// The CelestialObserver, which controls the camera.
  void                            setObserver(scene::CelestialObserver const& observer);
  scene::CelestialObserver&       getObserver();
  scene::CelestialObserver const& getObserver() const;

  /// Adds a CelestialAnchor to the SolarSystem.
  void registerAnchor(std::shared_ptr<scene::CelestialAnchor> const& anchor);

  /// Removes the CelestialAnchor from the SolarSystem.
  /// If the CelestialAnchor is also a CelestialBody and it was added via registerBody() it MUST be
  /// unregistered with unregisterBody() instead.
  void unregisterAnchor(std::shared_ptr<scene::CelestialAnchor> const& anchor);

  /// A list of all CelestialAnchors in the SolarSystem.
  std::set<std::shared_ptr<scene::CelestialAnchor>> const& getAnchors() const;

  /// @return The CelestialAnchor with the specified SPICE name or nullptr if it doesn't exist.
  std::shared_ptr<scene::CelestialAnchor> getAnchor(std::string const& sCenter) const;

  /// Adds a CelestialBody to the SolarSystem. A call to registerAnchor() is not needed, because
  /// it is done automatically.
  void registerBody(std::shared_ptr<scene::CelestialBody> const& body);

  /// Removes a CelestialBody from the SolarSystem. A call to unregisterAnchor() is not needed,
  /// because it is done automatically.
  void unregisterBody(std::shared_ptr<scene::CelestialBody> const& body);
  std::set<std::shared_ptr<scene::CelestialBody>> const& getBodies() const;
  std::shared_ptr<scene::CelestialBody>                  getBody(std::string const& sCenter) const;

  /// Updates all CelestialAnchors, the Sun and the CelestialObservers animations.
  void update();

  /// This scales the cs::scene::CelestialObserver of the solar system to move the
  /// closest body to a small world space distance. This distance depends on his or her *real*
  /// distance in outer space to the respective body.
  /// In order for the scientists to be able to interact with their environment, the next virtual
  /// celestial body must never be more than an arm’s length away. If the Solar System were always
  /// represented on a 1:1 scale, the virtual planetary surface would be too far away to work
  /// effectively with the simulation.
  /// As objects will be quite close to the observer in world space if the user is far away in
  /// *real* space, this also reduces the far clip distance in order to increase depth accuracy
  /// for objects close to the observer.
  void updateSceneScale();

  /// This method manages the SPICE frame changes when the observer moves from body to body. The
  /// active body is determined by its weight. The weight of a body is calculated by its size and
  /// distance to the observer.
  void updateObserverFrame();

  /// Gradually moves the observer's position and rotation from their current values to the given
  /// values.
  ///
  /// @param sCenter  The SPICE name of the targets center.
  /// @param sFrame   The SPICE reference frame of the targets location.
  /// @param position The target position in the targets coordinate system.
  /// @param rotation The target rotation in the targets coordinate system.
  /// @param duration The duration in Barycentric Dynamical Time to move to the new location.
  void flyObserverTo(std::string const& sCenter, std::string const& sFrame,
      glm::dvec3 const& position, glm::dquat const& rotation, double duration);

  /// Gradually moves the observer's position and rotation from their current values to the values
  /// matching the geographic coordinate system coordinates given.
  ///
  /// @param sCenter  The SPICE name of the targets center.
  /// @param sFrame   The SPICE reference frame of the targets location.
  /// @param lngLat   The target longitude and latitude in on the surface.
  /// @param height   The target height over the surface. (DocTODO see level or mean elevation?)
  /// @param duration The duration in Barycentric Dynamical Time to move to the new location.
  void flyObserverTo(std::string const& sCenter, std::string const& sFrame,
      glm::dvec2 const& lngLat, double height, double duration);

  /// Gradually moves the observer's position and rotation from their current values to a
  /// position in which the target object given by the SPICE frame center is in view.
  ///
  /// @param sCenter  The SPICE name of the targets center.
  /// @param sFrame   The SPICE reference frame of the targets location.
  /// @param duration The duration in Barycentric Dynamical Time to move to the new location.
  void flyObserverTo(std::string const& sCenter, std::string const& sFrame, double duration);

  /// DocTODO
  void setObserverToCamera();

  /// The methods below can be used for being notified about new CelestialBodies being added to or
  /// removed from the SolarSystem.
  uint64_t registerAddBodyListener(
      std::function<void(std::shared_ptr<scene::CelestialBody>)> listener);
  void unregisterAddBodyListener(uint64_t id);

  uint64_t registerRemoveBodyListener(
      std::function<void(std::shared_ptr<scene::CelestialBody>)> listener);
  void unregisterRemoveBodyListener(uint64_t id);

  // static utility functions ----------------------------------------------------------------------

  /// Initializes SPICE.
  void init(std::string const& sSpiceMetaFile);

  /// Return true, when init() has been called before.
  bool getIsInitialized() const;

  /// Cleans SPICE up.
  void deinit();

  /// For debugging purposes.
  /// Prints all loaded SPICE frames.
  static void printFrames();

  /// Disables SPICEs error reports.
  static void disableSpiceErrorReports();

  /// Enables SPICEs error reports.
  static void enableSpiceErrorReports();

  static void scaleRelativeToObserver(scene::CelestialAnchor& anchor,
      scene::CelestialObserver const& observer, double simulationTime, double baseDistance,
      double scaleFactor);
  static void turnToObserver(scene::CelestialAnchor& anchor,
      scene::CelestialObserver const& observer, double simulationTime, bool upIsNormal);

  /// Gives the radii to a given SPICE object.
  /// @param sCenterName The name of the SPICE object from which the radii are requested.
  static glm::dvec3 getRadii(std::string const& sCenterName);

  /// Generates a trail of points representing the given SPICE objects past movements.
  ///
  /// @param sCenterName  The SPICE name of the center, which the observer is currently locked to.
  /// @param sFrameName   The SPICE frame of reference.
  /// @param sTargetName  The SPICE name of the center of the body for whom the Trajectory shall
  ///                     be calculated.
  /// @param dStartTime   The start time of the trajectory. DocTODO more specific
  /// @param dEndTime     The end time of the trajectory. DocTODO more specific
  /// @param iSamples     The resolution of the trajectory.
  ///
  /// @return A vector of points, where the x, y and z values represent the position and the w
  ///         value represents the time of that point, when the body was at that location.
  static std::vector<glm::dvec4> calculateTrajectory(std::string const& sCenterName,
      std::string const& sFrameName, std::string const& sTargetName, double dStartTime,
      double dEndTime, int iSamples);

 private:
  std::shared_ptr<const Settings>                   mSettings;
  std::shared_ptr<utils::FrameTimings>              mFrameTimings;
  std::shared_ptr<GraphicsEngine>                   mGraphicsEngine;
  std::shared_ptr<TimeControl>                      mTimeControl;
  scene::CelestialObserver                          mObserver;
  std::shared_ptr<scene::CelestialObject>           mSun;
  std::set<std::shared_ptr<scene::CelestialAnchor>> mAnchors;
  std::set<std::shared_ptr<scene::CelestialBody>>   mBodies;

  bool mIsInitialized = false;

  uint64_t mListenerIds = 0;
  std::unordered_map<uint64_t, std::function<void(std::shared_ptr<scene::CelestialBody>)>>
      mAddBodyListeners;
  std::unordered_map<uint64_t, std::function<void(std::shared_ptr<scene::CelestialBody>)>>
      mRemoveBodyListeners;

  // These are used for measuring the observer speed.
  glm::dvec3                                     mLastPosition = glm::dvec3(0.0);
  std::chrono::high_resolution_clock::time_point mLastTime;
};

} // namespace cs::core

#endif // CS_CORE_SOLARSYSTEM_HPP
