////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "TimeInterval.hpp"

#include "../../../../../src/cs-utils/filesystem.hpp"
#include "../../../../../src/cs-utils/utils.hpp"
#include "../../../../csl-ogc/src/common/utils.hpp"
#include "../../../../../src/cs-core/TimeControl.hpp"

namespace csp::visualquery {

////////////////////////////////////////////////////////////////////////////////////////////////////

const std::string TimeInterval::sName = "TimeInterval";

////////////////////////////////////////////////////////////////////////////////////////////////////

std::string TimeInterval::sSource() {
  return cs::utils::filesystem::loadToString(
      "../share/resources/nodes/csp-visual-query/TimeInterval.js");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::unique_ptr<TimeInterval> TimeInterval::sCreate(std::shared_ptr<cs::core::TimeControl> timeControl) {
  return std::make_unique<TimeInterval>(timeControl);
}

TimeInterval::TimeInterval(std::shared_ptr<cs::core::TimeControl> timeControl) 
  : mTimeControl(std::move(timeControl))
  , mValue(std::string())
  , mIntervals(std::vector<csl::ogc::TimeInterval>())
  , mSelectedIntervalIndex(-1)
  , mTimeOperationCounter(0)
  , mMaxTimeOperationCounter(0)
  , mSyncSimTime(false) 
  , mTimeConnection(0) {

  mTimeConnection = mTimeControl->pSimulationTime.connect([this](double) { process(); });
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TimeInterval::~TimeInterval() {
  mTimeControl->pSimulationTime.disconnect(mTimeConnection);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::string const& TimeInterval::getName() const {
  return sName;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void TimeInterval::process() {
  auto intervals = readInput<std::vector<csl::ogc::TimeInterval>>("timeIntervalsIn", std::vector<csl::ogc::TimeInterval>());

  // intervals input changed
  if (mIntervals != intervals) {

    mIntervals = intervals;
    mSelectedIntervalIndex = -1; // "none" preselected
    sendMessageToJS(createIntervalsMessage());
  }

  // check if node time is synchronous with simulation time
  if (mSyncSimTime) {
    boost::posix_time::ptime simTime = cs::utils::convert::time::toPosix(mTimeControl->pSimulationTime.get());

    // if no interval selected: write the current simulation time to the output
    if (mSelectedIntervalIndex == -1) {
      mValue = boost::posix_time::to_iso_extended_string(simTime);

      // send new time to js
      nlohmann::json newSelectedTime;
      newSelectedTime["currentTime"] = mValue;
      sendMessageToJS(newSelectedTime);

      writeOutput("value", mValue);
      return;
    }

    // if interval selected: find correct time step for the current simulations time
    boost::posix_time::ptime currentNodeTime = csl::ogc::utils::addDurationToTime(mIntervals[mSelectedIntervalIndex].mStartTime, 
      mIntervals[mSelectedIntervalIndex].mSampleDuration, mTimeOperationCounter);
    boost::posix_time::ptime nextNodeTime = csl::ogc::utils::addDurationToTime(mIntervals[mSelectedIntervalIndex].mStartTime, 
      mIntervals[mSelectedIntervalIndex].mSampleDuration, (mTimeOperationCounter < mMaxTimeOperationCounter ? mTimeOperationCounter + 1 : mTimeOperationCounter));

    // check if current time step is valid for the simulation time
    if (simTime >= currentNodeTime && simTime < nextNodeTime) {
      // simulation time is inside the current time step
      // no need to do anything

    // if simulation time is not inside current time step: find closest time step and set as the new one
    } else {
      if (simTime > currentNodeTime) {        
        while (true) {
          if (mTimeOperationCounter == mMaxTimeOperationCounter) {
            // if the simulation time is greater then last time step -> last time steps gets used
            break;
          }
          mTimeOperationCounter++;

          // if the simulation time is smaller than the next step then the new current step is the previous one
          boost::posix_time::ptime nextTime = csl::ogc::utils::addDurationToTime(mIntervals[mSelectedIntervalIndex].mStartTime, 
            mIntervals[mSelectedIntervalIndex].mSampleDuration, mTimeOperationCounter);
          if (simTime < nextTime) {
            mTimeOperationCounter--;
            break;
          }
        }
      
      } else {
        while (true) {
          // if the simulation time is smaller then last time step -> first time steps gets used
          if (mTimeOperationCounter == 0) {
            break;
          }
          mTimeOperationCounter--;

          // if the simulation time is greater than the previous step then the new current step is the previous one
          boost::posix_time::ptime prevTime = csl::ogc::utils::addDurationToTime(mIntervals[mSelectedIntervalIndex].mStartTime, 
            mIntervals[mSelectedIntervalIndex].mSampleDuration, mTimeOperationCounter);
          if (simTime > prevTime) {
            break;
          }
        }
      }
      // set new output value
      boost::posix_time::ptime newTime = csl::ogc::utils::addDurationToTime(mIntervals[mSelectedIntervalIndex].mStartTime, 
            mIntervals[mSelectedIntervalIndex].mSampleDuration, mTimeOperationCounter);
      mValue = boost::posix_time::to_iso_extended_string(newTime);
      
      // send new time step to js
      nlohmann::json newSelectedTime;
      newSelectedTime["currentTime"] = mValue;
      sendMessageToJS(newSelectedTime);
    }
  }

  writeOutput("value", mValue);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void TimeInterval::onMessageFromJS(nlohmann::json const& message) {
  // The message sent via CosmoScout.sendMessageToCPP() contains the entered date and time.

  // new interval selected
  if (message.find("intervalIndex") != message.end()) {
    mSelectedIntervalIndex = message["intervalIndex"];

    // "None" selected    
    if (mSelectedIntervalIndex == -1) {
      mTimeOperationCounter = 0;
      mMaxTimeOperationCounter = 0;
      mValue = "";
    
    // interval selected
    } else {
      // start time of interval gets automatically selected as new selected time point
      mValue = boost::posix_time::to_iso_extended_string(mIntervals[mSelectedIntervalIndex].mStartTime);
      if (mSyncSimTime) {
        mTimeControl->setTime(cs::utils::convert::time::toSpice(mIntervals[mSelectedIntervalIndex].mStartTime));
      }
      
      // send newly set time point to js
      nlohmann::json newSelectedTime;
      newSelectedTime["currentTime"] = mValue;
      sendMessageToJS(newSelectedTime);

      // compute max number of steps in time interval
      mMaxTimeOperationCounter = 0;
      while (csl::ogc::utils::addDurationToTime(mIntervals[mSelectedIntervalIndex].mStartTime, 
          mIntervals[mSelectedIntervalIndex].mSampleDuration, mMaxTimeOperationCounter) < mIntervals[mSelectedIntervalIndex].mEndTime) {  
        mMaxTimeOperationCounter++;
      }
    }
  }

  // time step change:
  if (message.find("timeOperation") != message.end()) {

    boost::posix_time::ptime newTime;
    bool validOperation = true;

    if (message["timeOperation"] == "first") {
      mTimeOperationCounter = 0;
      newTime = mIntervals[mSelectedIntervalIndex].mStartTime;
    }

    if (message["timeOperation"] == "prev") {
      if (mTimeOperationCounter - 1 >= 0) {
        mTimeOperationCounter--;
        newTime = csl::ogc::utils::addDurationToTime(mIntervals[mSelectedIntervalIndex].mStartTime, 
          mIntervals[mSelectedIntervalIndex].mSampleDuration, mTimeOperationCounter);
      } else {
        validOperation = false;
      }
    }

    if (message["timeOperation"] == "next") {
      if (mTimeOperationCounter + 1 <= mMaxTimeOperationCounter) {
        mTimeOperationCounter++;
        newTime = csl::ogc::utils::addDurationToTime(mIntervals[mSelectedIntervalIndex].mStartTime, 
          mIntervals[mSelectedIntervalIndex].mSampleDuration, mTimeOperationCounter);
      } else {
        validOperation = false;
      }
    }

    if (message["timeOperation"] == "last") {
      mTimeOperationCounter = mMaxTimeOperationCounter;
      newTime = mIntervals[mSelectedIntervalIndex].mEndTime;
    }
    
    if (validOperation) {
      mValue = boost::posix_time::to_iso_extended_string(newTime);
      if (mSyncSimTime) {
        mTimeControl->setTime(cs::utils::convert::time::toSpice(newTime));
      }

      nlohmann::json newSelectedTime;
      newSelectedTime["currentTime"] = mValue;
      sendMessageToJS(newSelectedTime);
    }
  }

  // change of sync with simulation time 
  if (message.find("syncSimTime") != message.end()) {
    mSyncSimTime = message["syncSimTime"];

    if (mSyncSimTime && mSelectedIntervalIndex != -1) {
      // set sim time to currently selected time step
      mTimeControl->setTime(cs::utils::convert::time::toSpice(
        csl::ogc::utils::addDurationToTime(mIntervals[mSelectedIntervalIndex].mStartTime, 
          mIntervals[mSelectedIntervalIndex].mSampleDuration, mTimeOperationCounter)));

    } else {
      // no interval selected: reset the display time 
      if (mSelectedIntervalIndex == -1) {
        mValue = "";
        nlohmann::json newSelectedTime;
        newSelectedTime["currentTime"] = "reset";
        sendMessageToJS(newSelectedTime);
      }
    }
  }
  process();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

nlohmann::json TimeInterval::getData() const {
  nlohmann::json data;
  data["currentTime"] = mValue;
  data["selectedIntervalIndex"] = mSelectedIntervalIndex;
  data["timeOperationCounter"] = mTimeOperationCounter;
  data["maxTimeOperationCounter"] = mMaxTimeOperationCounter;
  data["syncSimTime"] = mSyncSimTime;
  
  data["intervals"] = createIntervalsMessage();
  return data;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void TimeInterval::setData(nlohmann::json const& json) {
  mValue = (json["value"]);
  mSelectedIntervalIndex = (json["selectedIntervalIndex"]);
  mTimeOperationCounter = (json["timeOperationCounter"]);
  mMaxTimeOperationCounter = (json["maxTimeOperationCounter"]);
  mSyncSimTime = (json["syncSimTime"]);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

nlohmann::json TimeInterval::createIntervalsMessage() const {
  nlohmann::json message; 

  if (mIntervals.empty()) {
    message["intervals"] = "reset";
    
  } else {
    message["intervals"] = nlohmann::json::array();;

    for (auto interval : mIntervals) {
      nlohmann::json intervalJson;
      intervalJson["start"] = boost::posix_time::to_iso_extended_string(interval.mStartTime);
      intervalJson["end"] = boost::posix_time::to_iso_extended_string(interval.mEndTime);
      
      auto duration = interval.mSampleDuration;
      if (duration.mYears) {
        intervalJson["step"]["size"] = duration.mYears;
        intervalJson["step"]["unit"] = duration.mYears > 1 ? "years" : "year";
      
      } else if (duration.mMonths) {
        intervalJson["step"]["size"] = duration.mMonths;
        intervalJson["step"]["unit"] = duration.mMonths > 1 ? "months" : "month";
      
      } else {
        intervalJson["step"]["size"] = duration.mTimeDuration.total_seconds();
        intervalJson["step"]["unit"] = "sec";
      }

      message["intervals"].insert(message["intervals"].end(), intervalJson);
    }
  }

  return message;
}

} // namespace csp::visualquery
