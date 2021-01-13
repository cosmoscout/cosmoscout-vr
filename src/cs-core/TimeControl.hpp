////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CS_CORE_TIME_CONTROL_HPP
#define CS_CORE_TIME_CONTROL_HPP

#include "cs_core_export.hpp"

#include "../cs-utils/AnimatedValue.hpp"
#include "../cs-utils/Property.hpp"

#include <memory>

namespace cs::core {
class Settings;

/// The keeper of time. You can control the simulation time with this class. You can
/// accelerate/decelerate the flow of time or jump to specific points in time.
/// All time units are in Barycentric Dynamical Time (TDB).
class CS_CORE_EXPORT TimeControl {
 public:
  /// The current time in TDB. Consider this to be read-only.
  utils::Property<double> pSimulationTime = 0.0;

  /// The current speed of the simulation. Consider this to be read-only.
  utils::Property<float> pTimeSpeed = 1.F;

  explicit TimeControl(std::shared_ptr<Settings> settings);

  TimeControl(TimeControl const& other) = delete;
  TimeControl(TimeControl&& other)      = delete;

  TimeControl& operator=(TimeControl const& other) = delete;
  TimeControl& operator=(TimeControl&& other) = delete;

  ~TimeControl();

  /// Updates the simulation time based on the current time speed. This is called once a frame by
  /// the application class, there is no need to call this somewhere else.
  void update();

  /// Set the simulation time to a specific point in time. The TimeControl class smoothly transition
  /// to that point in time, if it is closer to the current simulation time than the given
  /// threshold. The given target time will be clamped to the minimum and maximum date specified in
  /// the settings given at construction time.
  /// @param time      The target time in TDB.
  /// @param duration  The animation time in seconds. There will be only an animation if the
  ///                  absolute difference between the current simulation and the target time is
  ///                  less than the given threshold. The duration will be shortened for smaller
  ///                  differences.
  /// @param threshold In seconds. If the absolute difference between simulation time and target
  ///                  time exceeds this threshold, no transition will be made.
  void setTime(double tTime, double duration = 0.0, double threshold = 48.0 * 60.0 * 60.0);

  /// Resets the simulation time to the starting time or to the current time depending on the
  /// startup settings defined in the configuration file, where a value of "today" will result in
  /// the current time. The TimeControl class smoothly transition to that point in time, if it is
  /// closer to the current simulation time than the given threshold.
  /// @param duration  The animation time in seconds. There will be only an animation if the
  ///                  absolute difference between the current simulation and the target time is
  ///                  less than the given threshold. The duration will be shortened for smaller
  ///                  differences.
  /// @param threshold In seconds. If the absolute difference between simulation time and target
  ///                  time exceeds this threshold, no transition will be made.
  void resetTime(double duration = 0.0, double threshold = 48.0 * 60.0 * 60.0);

  /// Set the time speed to a specific value. If set to zero, the simulation will be paused.
  /// @param speed  The new time speed.
  void setTimeSpeed(float speed);

 private:
  bool   mInitialized = false;
  double mLastUpdate  = 0.0;
  double pMaxDate     = 0.0;
  double pMinDate     = 0.0;

  utils::AnimatedValue<double> mAnimatedTime;
  bool                         mAnimationInProgress = false;

  std::shared_ptr<Settings> mSettings;
};

} // namespace cs::core

#endif // CS_CORE_TIME_CONTROL_HPP
