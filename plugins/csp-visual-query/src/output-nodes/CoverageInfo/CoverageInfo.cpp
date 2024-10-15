////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "CoverageInfo.hpp"

#include "../../../../../src/cs-utils/filesystem.hpp"
#include "../../types/CoverageContainer.hpp"

namespace csp::visualquery {

////////////////////////////////////////////////////////////////////////////////////////////////////

const std::string CoverageInfo::sName = "CoverageInfo";

////////////////////////////////////////////////////////////////////////////////////////////////////

std::string CoverageInfo::sSource() {
  return cs::utils::filesystem::loadToString(
      "../share/resources/nodes/csp-visual-query/CoverageInfo.js");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::unique_ptr<CoverageInfo> CoverageInfo::sCreate() {
  return std::make_unique<CoverageInfo>();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::string const& CoverageInfo::getName() const {
  return sName;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void CoverageInfo::process() {
  auto           coverage = readInput<std::shared_ptr<CoverageContainer>>("coverageIn", nullptr);
  nlohmann::json json;

  if (coverage != nullptr) {
    auto coverageSettings = coverage->mImageChannel->getSettings();

    json["bounds"]["minLong"] = coverageSettings.mBounds.mMinLon;
    json["bounds"]["maxLong"] = coverageSettings.mBounds.mMaxLon;
    json["bounds"]["minLat"]  = coverageSettings.mBounds.mMinLat;
    json["bounds"]["maxLat"]  = coverageSettings.mBounds.mMaxLat;

    if (coverageSettings.mAxisResolution[0] > 0 && coverageSettings.mAxisResolution[1] > 0) {
      json["size"]["width"]  = coverageSettings.mAxisResolution[0];
      json["size"]["height"] = coverageSettings.mAxisResolution[1];
    }

    json["layers"] = coverageSettings.mNumLayers;

    if (coverageSettings.mAttribution.has_value()) {
      json["attribution"] = coverageSettings.mAttribution.value();
    }

    std::vector<std::string> intervals;
    for (auto interval : coverageSettings.mTimeIntervals) {
      std::string intervalString = boost::posix_time::to_iso_extended_string(interval.mStartTime);
      intervalString.append(" - ");
      intervalString.append(boost::posix_time::to_iso_extended_string(interval.mEndTime));
      intervals.push_back(intervalString);
    }

    if (intervals.size() > 0) {
      json["intervals"] = intervals;
    }

    auto keywords = coverage->mImageChannel->getKeywords();
    if (keywords.has_value()) {
      json["keywords"] = keywords.value();
    }

    auto abstract = coverage->mImageChannel->getAbstract();
    if (abstract.has_value()) {
      json["abstract"] = abstract.value();
    }
  }
  sendMessageToJS(json);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void CoverageInfo::onMessageFromJS(nlohmann::json const& message) {
}

////////////////////////////////////////////////////////////////////////////////////////////////////

nlohmann::json CoverageInfo::getData() const {
  return nlohmann::json();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void CoverageInfo::setData(nlohmann::json const& json) {
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::visualquery
