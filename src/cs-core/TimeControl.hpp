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
  /// The current time in TDB.
  utils::Property<double> pSimulationTime = 0.0;

  /// The current speed of the simulation.
  utils::Property<float> pTimeSpeed = 1.f;

  TimeControl(std::shared_ptr<const Settings> const& settings);
  ~TimeControl();

  /// Updates the time every update. No need to call this.
  void update();

  /// Set the simulation time to a specific point in time. The TimeControl class tries to
  /// transition to that point in time smoothly, if it is close to the current simulation time.
  /// @param time The target time in TDB.
  void setTime(double tTime);

  /// Set the simulation time to a specific point in time. The transition is not done smoothly,
  /// even if the piont is close to the current simulation time.
  /// @param The target time in TDB.
  void setTimeWithoutAnimation(double tTime);

  /// Resets the simulation time to the starting time or to the current time depending on the
  /// startup settings defined in the configuration file, where a value of "today" will result in
  /// the current time.
  void resetTime();

  /// Double the passage of time. Gotta go fast.
  void increaseTimeSpeed();

  /// Half the passage of time. No need to hurry.
  void decreaseTimeSpeed();

  /// Set the time speed to a specific value
  /// @param The new time speed
  void setTimeSpeed(float speed);

 private:
  double mLastUpdate = -1.0;

  utils::AnimatedValue<double> mAnimatedTime;
  bool                         mAnimationInProgress = false;

  std::shared_ptr<const Settings> mSettings;
};

} // namespace cs::core

#endif // CS_CORE_TIME_CONTROL_HPP
