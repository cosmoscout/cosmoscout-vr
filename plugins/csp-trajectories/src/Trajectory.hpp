////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CSP_TRAJECTORIES_TRAJECTORY_HPP
#define CSP_TRAJECTORIES_TRAJECTORY_HPP

#include "Plugin.hpp"

#include "../../../src/cs-scene/CelestialObject.hpp"
#include "../../../src/cs-scene/Trajectory.hpp"

#include <VistaBase/VistaColor.h>
#include <VistaKernel/GraphicsManager/VistaOpenGLDraw.h>
#include <memory>

namespace csp::trajectories {

/// A trajectory trails behind an object in space to give a better understanding of its movement.
class Trajectory : public cs::scene::CelestialObject, public IVistaOpenGLDraw {
 public:
  /// The length of the trajectory in days.
  cs::utils::Property<double> pLength = 1.0;

  /// The trajectory is drawn using this many linear pieces.
  cs::utils::Property<uint32_t> pSamples = 100;

  /// The color of the trajectory.
  cs::utils::Property<glm::vec3> pColor = glm::vec3(1, 1, 1);

  Trajectory(std::shared_ptr<Plugin::Settings> pluginSettings, std::string sTargetCenter,
      std::string sTargetFrame, std::string const& sParentCenter, std::string const& sParentFrame,
      double tStartExistence, double tEndExistence);

  Trajectory(Trajectory const& other) = delete;
  Trajectory(Trajectory&& other)      = delete;

  Trajectory& operator=(Trajectory const& other) = delete;
  Trajectory& operator=(Trajectory&& other) = delete;

  ~Trajectory() override;

  /// This is called automatically by the SolarSystem.
  void update(double tTime, cs::scene::CelestialObserver const& oObs) override;

  /// The trajectory visualizes the path of this body.
  void               setTargetCenterName(std::string const& sCenterName);
  void               setTargetFrameName(std::string const& sFrameName);
  std::string const& getTargetCenterName() const;
  std::string const& getTargetFrameName() const;

  /// The trajectory is drawn relative to this body.
  void setCenterName(std::string const& sCenterName) override;
  void setFrameName(std::string const& sFrameName) override;

  bool Do() override;
  bool GetBoundingBox(VistaBoundingBox& bb) override;

 private:
  std::shared_ptr<Plugin::Settings> mPluginSettings;
  cs::scene::Trajectory             mTrajectory;

  std::unique_ptr<VistaOpenGLNode> mGLNode;

  std::string             mTargetCenter;
  std::string             mTargetFrame;
  std::vector<glm::dvec4> mPoints;
  int                     mStartIndex;
  double                  mLastSampleTime{};
  double                  mLastUpdateTime;
  double                  mLastFrameTime{};

  bool mTrailIsInExistence = false;
};

} // namespace csp::trajectories

#endif // CSP_TRAJECTORIES_TRAJECTORY_HPP
