////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "TimeInterval.hpp"

#include "../../../../../src/cs-utils/filesystem.hpp"
#include "../../../../../src/cs-utils/utils.hpp"
#include "../../../../csl-ogc/src/common/utils.hpp"

namespace csp::visualquery {

////////////////////////////////////////////////////////////////////////////////////////////////////

const std::string TimeInterval::sName = "TimeInterval";

////////////////////////////////////////////////////////////////////////////////////////////////////

std::string TimeInterval::sSource() {
  return cs::utils::filesystem::loadToString(
      "../share/resources/nodes/csp-visual-query/TimeInterval.js");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::unique_ptr<TimeInterval> TimeInterval::sCreate() {
  return std::make_unique<TimeInterval>();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::string const& TimeInterval::getName() const {
  return sName;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void TimeInterval::process() {
  auto intervals = readInput<std::vector<csl::ogc::TimeInterval>>("timeIntervalsIn", std::vector<csl::ogc::TimeInterval>());

  // send new intervals
  // if (intervals != mIntervals) {
    mIntervals = intervals;
    std::vector<std::string> intervalNames;
    
    for (auto interval : mIntervals) {
      intervalNames.push_back(
        boost::posix_time::to_iso_extended_string(interval.mStartTime) + "/" +
        boost::posix_time::to_iso_extended_string(interval.mEndTime)
      );
    }
    nlohmann::json message;
    message["intervals"] = intervalNames;
    sendMessageToJS(message);
  //  return;
  // }


  writeOutput("value", mValue);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void TimeInterval::onMessageFromJS(nlohmann::json const& message) {
  // The message sent via CosmoScout.sendMessageToCPP() contains the entered date and time.
  mValue = message;

  // Whenever the user entered a value, we write it to the output socket by calling the process()
  // method. Writing the output will not trigger a graph reprocessing right away, it will only queue
  // up the connected nodes for being processed in the next update step (and only if the value
  // actually changed).
  process();
}



////////////////////////////////////////////////////////////////////////////////////////////////////

nlohmann::json TimeInterval::getData() const {
  return {{"value", mValue}};
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void TimeInterval::setData(nlohmann::json const& json) {
  mValue = (json["value"]);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::visualquery
