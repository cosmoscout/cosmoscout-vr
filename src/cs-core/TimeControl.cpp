////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "TimeControl.hpp"

#include "../cs-utils/convert.hpp"
#include "Settings.hpp"
#include "logger.hpp"

#include <VistaKernel/VistaSystem.h>
#include <utility>

namespace cs::core {

////////////////////////////////////////////////////////////////////////////////////////////////////

TimeControl::TimeControl(std::shared_ptr<core::Settings> settings)
    : mSettings(std::move(settings)) {

  // Update the mStartDate in the settings. If the current simulation time differs less than one
  // minute from the current system time, we write "today", else the actual simulation date.
  mSettings->onSave().connect([this]() {
    auto now = utils::convert::time::toSpice(boost::posix_time::microsec_clock::universal_time());
    if (std::abs(pSimulationTime.get() - now) < 60) {
      mSettings->mStartDate = "today";
    } else {
      mSettings->mStartDate = utils::convert::time::toString(pSimulationTime.get());
    }
  });

  mSettings->onLoad().connect([this]() {
    if (mSettings->mStartDate == "today") {
      setTime(
          utils::convert::time::toSpice(boost::posix_time::microsec_clock::universal_time()), 5.0);
    } else {
      try {
        setTime(utils::convert::time::toSpice(mSettings->mStartDate), 5.0);
      } catch (std::exception const&) {
        throw std::runtime_error("Could not parse the 'startDate' setting. It should either be "
                                 "'today' or in the format '1969-07-20T20:17:40.000Z'.");
      }
    }
  });

  // Tell the user what's going on.
  logger().debug("Creating TimeControl.");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TimeControl::~TimeControl() {
  // Tell the user what's going on.
  logger().debug("Deleting TimeControl.");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void TimeControl::update() {

  double now = utils::convert::time::toSpice(boost::posix_time::microsec_clock::universal_time());

  // Initialize our members. This has to be done here as SPICE is not yet loaded at construction
  // time.
  if (!mInitialized) {
    mSettings->pMaxDate.connectAndTouch(
        [this](std::string const& val) { pMaxDate = utils::convert::time::toSpice(val); });

    mSettings->pMinDate.connectAndTouch(
        [this](std::string const& val) { pMinDate = utils::convert::time::toSpice(val); });

    if (mSettings->mStartDate == "today") {
      setTime(utils::convert::time::toSpice(boost::posix_time::microsec_clock::universal_time()));
    } else {
      try {
        setTime(utils::convert::time::toSpice(mSettings->mStartDate));
      } catch (std::exception const&) {
        throw std::runtime_error("Could not parse the 'startDate' setting. It should either be "
                                 "'today' or in the format '1969-07-20T20:17:40.000Z'.");
      }
    }

    mLastUpdate = now;

    mInitialized = true;
  }

  if (mAnimationInProgress) {
    pSimulationTime = mAnimatedTime.get(now);

    if (mAnimatedTime.mEndTime < now) {
      mAnimationInProgress = false;
    }
  } else {
    double tTime = pSimulationTime.get() + (now - mLastUpdate) * pTimeSpeed.get();
    if (tTime >= pMaxDate || tTime <= pMinDate) {
      pSimulationTime = std::clamp(tTime, pMinDate, pMaxDate);
      setTimeSpeed(0);
    } else {
      pSimulationTime = tTime;
    }
  }

  mLastUpdate = now;
} // namespace cs::core

////////////////////////////////////////////////////////////////////////////////////////////////////

void TimeControl::setTime(double tTime, double duration, double threshold) {
  double now = utils::convert::time::toSpice(boost::posix_time::microsec_clock::universal_time());
  double difference = std::abs(pSimulationTime.get() - tTime);

  if (tTime >= pMaxDate || tTime <= pMinDate) {
    pSimulationTime = std::clamp(tTime, pMinDate, pMaxDate);
    setTimeSpeed(0);
  } else if (duration <= 0.0 || difference > std::abs(threshold) || threshold <= 0) {
    // Make no animation for very large time changes.
    pSimulationTime = tTime;
  } else {
    double const reduction = 0.2;
    double const inverse   = 1.0 - reduction;

    // Make smooth animation for time changes greater than the given threshold. We reduce the
    // duration up to 20% of the given value if the difference is smaller than the threshold.
    duration = reduction * duration + inverse * duration * difference / threshold;

    mAnimatedTime = utils::AnimatedValue<double>(
        pSimulationTime.get(), tTime, now, now + duration, utils::AnimationDirection::eInOut);
    mAnimationInProgress = true;
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void TimeControl::resetTime(double duration, double threshold) {

  if (mSettings->mResetDate == "today") {
    setTime(utils::convert::time::toSpice(boost::posix_time::microsec_clock::universal_time()),
        duration, threshold);
  } else {
    try {
      setTime(utils::convert::time::toSpice(mSettings->mResetDate), duration, threshold);
    } catch (std::exception const&) {
      throw std::runtime_error("Could not parse the 'resetDate' setting. It should either be "
                               "'today' or in the format '1969-07-20T20:17:40.000Z'.");
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void TimeControl::setTimeSpeed(float speed) {
  pTimeSpeed = speed;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cs::core
