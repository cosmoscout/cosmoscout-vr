////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

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
class Trajectory : public IVistaOpenGLDraw {
 public:
  /// The length of the trajectory in days.
  cs::utils::Property<double> pLength = 1.0;

  /// The trajectory is drawn using this many linear pieces.
  cs::utils::Property<uint32_t> pSamples = 100;

  /// The color of the trajectory.
  cs::utils::Property<glm::vec3> pColor = glm::vec3(1, 1, 1);

  Trajectory(std::shared_ptr<Plugin::Settings> pluginSettings,
      std::shared_ptr<cs::core::SolarSystem>   solarSystem);

  Trajectory(Trajectory const& other) = delete;
  Trajectory(Trajectory&& other)      = delete;

  Trajectory& operator=(Trajectory const& other) = delete;
  Trajectory& operator=(Trajectory&& other)      = delete;

  ~Trajectory() override;

  /// This is called by the Plugin.
  void update(double tTime);

  /// The trajectory visualizes the path of this body.
  void               setTargetName(std::string objectName);
  std::string const& getTargetName() const;

  /// The trajectory is drawn relative to this body.
  void               setParentName(std::string objectName);
  std::string const& getParentName() const;

  bool Do() override;
  bool GetBoundingBox(VistaBoundingBox& bb) override;

 private:
  std::shared_ptr<Plugin::Settings>      mPluginSettings;
  std::shared_ptr<cs::core::SolarSystem> mSolarSystem;
  cs::scene::Trajectory                  mTrajectory;

  std::unique_ptr<VistaOpenGLNode> mGLNode;

  std::string mTargetName;
  std::string mParentName;

  std::vector<glm::dvec4> mPoints;
  int                     mStartIndex     = 0;
  double                  mLastSampleTime = 0.0;
  double                  mLastUpdateTime = -1.0;
  double                  mLastFrameTime  = 0.0;
};

} // namespace csp::trajectories

#endif // CSP_TRAJECTORIES_TRAJECTORY_HPP
