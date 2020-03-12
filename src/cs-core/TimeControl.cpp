////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "TimeControl.hpp"

#include "../cs-utils/convert.hpp"
#include "Settings.hpp"

#include <VistaKernel/VistaSystem.h>
#include <spdlog/spdlog.h>

namespace cs::core {

////////////////////////////////////////////////////////////////////////////////////////////////////

TimeControl::TimeControl(std::shared_ptr<const core::Settings> const& settings)
    : mSettings(settings) {

  // Tell the user what's going on.
  spdlog::debug("Creating TimeControl.");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TimeControl::~TimeControl() {
  // Tell the user what's going on.
  spdlog::debug("Deleting TimeControl.");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void TimeControl::update() {
  // Initialize our members. This has to be done here as SPICE is not yet loaded at construction
  // time.
  if (mStartDate == "") {
    mStartDate = mSettings->mStartDate;
    mMaxDate =
        utils::convert::toSpiceTime(boost::posix_time::time_from_string(mSettings->mMaxDate));
    mMinDate =
        utils::convert::toSpiceTime(boost::posix_time::time_from_string(mSettings->mMinDate));
  }

  double now = utils::convert::toSpiceTime(boost::posix_time::microsec_clock::universal_time());

  if (mLastUpdate < 0.0) {
    resetTime();
    mLastUpdate = now;
  }

  if (mAnimationInProgress) {
    pSimulationTime = mAnimatedTime.get(now);

    if (mAnimatedTime.mEndTime < now) {
      mAnimationInProgress = false;
    }
  } else {
    double tTime = pSimulationTime.get() + (now - mLastUpdate) * pTimeSpeed.get();
    if (tTime >= mMaxDate || tTime <= mMinDate) {
      pSimulationTime = std::clamp(tTime, mMinDate, mMaxDate);
      setTimeSpeed(0);
    } else {
      pSimulationTime = tTime;
    }
  }

  mLastUpdate = now;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void TimeControl::setTime(double tTime, double duration, double threshold) {
  double now = utils::convert::toSpiceTime(boost::posix_time::microsec_clock::universal_time());
  double difference = std::abs(pSimulationTime.get() - tTime);

  if (tTime >= mMaxDate || tTime <= mMinDate) {
    pSimulationTime = std::clamp(tTime, mMinDate, mMaxDate);
    setTimeSpeed(0);
  } else if (duration <= 0.0 || difference > std::abs(threshold) || threshold <= 0) {
    // Make no animation for very large time changes.
    pSimulationTime = tTime;
  } else {
    // Make smooth animation for time changes greater than the given threshold. We reduce the
    // duration up to 20% of the given value if the difference is smaller than the threshold.
    duration = 0.2 * duration + 0.8 * duration * difference / threshold;

    mAnimatedTime = utils::AnimatedValue<double>(
        pSimulationTime.get(), tTime, now, now + duration, utils::AnimationDirection::eInOut);
    mAnimationInProgress = true;
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void TimeControl::resetTime(double duration, double threshold) {

  double tTime;

  if (mStartDate == "today") {
    tTime = utils::convert::toSpiceTime(boost::posix_time::microsec_clock::universal_time());
  } else {
    try {
      tTime = utils::convert::toSpiceTime(boost::posix_time::time_from_string(mStartDate));
    } catch (std::exception const& e) {
      throw std::runtime_error("Could not parse the 'startDate' setting. It should either be "
                               "'today' or in the format '1969-07-20 20:17:40.000'.");
    }
  }

  setTime(tTime, duration, threshold);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void TimeControl::setTimeSpeed(float speed) {
  pTimeSpeed = speed;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cs::core
