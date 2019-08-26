////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "TimeControl.hpp"

#include "../cs-utils/convert.hpp"
#include "Settings.hpp"

#include <VistaKernel/VistaSystem.h>

namespace cs::core {

////////////////////////////////////////////////////////////////////////////////////////////////////

TimeControl::TimeControl(std::shared_ptr<const core::Settings> const& settings)
    : mSettings(settings) {
  resetTime();
  mLastUpdate = utils::convert::toSpiceTime(boost::posix_time::microsec_clock::universal_time());
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void TimeControl::update() {
  double now = utils::convert::toSpiceTime(boost::posix_time::microsec_clock::universal_time());

  if (mAnimationInProgress) {
    pSimulationTime = mAnimatedTime.get(now);

    if (mAnimatedTime.mEndTime < now) {
      mAnimationInProgress = false;
    }
  } else {
    pSimulationTime = pSimulationTime.get() + (now - mLastUpdate) * pTimeSpeed.get();
  }

  mLastUpdate = now;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void TimeControl::setTime(double tTime) {
  double now = utils::convert::toSpiceTime(boost::posix_time::microsec_clock::universal_time());

  double step = std::abs(pSimulationTime.get() - tTime) / 60 / 60;

  if (step > 48) {
    // Make no animation for very large time changes.
    pSimulationTime = tTime;
  } else {
    // Make smooth animation for time changes.
    double duration = 0.5;

    // Increase duration for longer time steps.
    duration += glm::clamp(step / 20.0, 0.0, 5.0);

    mAnimatedTime = utils::AnimatedValue<double>(
        pSimulationTime.get(), tTime, now, now + duration, utils::AnimationDirection::eInOut);
    mAnimationInProgress = true;
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void TimeControl::resetTime() {
  std::string startDate = mSettings->mStartDate;

  double tTime;

  if (startDate == "today") {
    tTime = utils::convert::toSpiceTime(boost::posix_time::microsec_clock::universal_time());
  } else {
    try {
      tTime = utils::convert::toSpiceTime(boost::posix_time::time_from_string(startDate));
    } catch (std::exception const& e) {
      throw std::runtime_error("Could not parse the 'startDate' setting. It should either be 'today' or in the format 'YYYY-MM-DD HH:mm:ss.fff'.");
    }
  }

  setTime(tTime);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void TimeControl::increaseTimeSpeed() {
  float speed(pTimeSpeed.get());

  if (speed > 0.f) {
    speed *= 2.f;
  } else if (speed <= -1.f) {
    speed /= 2.f;
  } else {
    speed = 0.5f;
  }

  pTimeSpeed = speed;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void TimeControl::decreaseTimeSpeed() {
  float speed(pTimeSpeed.get());

  if (speed >= 1.f) {
    speed /= 2.f;
  } else if (speed < 0.f) {
    speed *= 2.f;
  } else {
    speed = -0.5f;
  }

  pTimeSpeed = speed;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cs::core
