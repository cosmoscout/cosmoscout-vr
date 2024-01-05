////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "CoverageViewer.hpp"

#include "../../../../../src/cs-utils/filesystem.hpp"
#include "../../types/CoverageContainer.hpp"

namespace csp::visualquery {

////////////////////////////////////////////////////////////////////////////////////////////////////

const std::string CoverageViewer::sName = "CoverageViewer";

////////////////////////////////////////////////////////////////////////////////////////////////////

std::string CoverageViewer::sSource() {
  return cs::utils::filesystem::loadToString(
      "../share/resources/nodes/csp-visual-query/CoverageViewer.js");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::unique_ptr<CoverageViewer> CoverageViewer::sCreate() {
  return std::make_unique<CoverageViewer>();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::string const& CoverageViewer::getName() const {
  return sName;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void CoverageViewer::process() {
  auto coverage = readInput<std::shared_ptr<CoverageContainer>>("coverageIn", nullptr);
  nlohmann::json json;

  if (coverage != nullptr) {
    auto coverageSettings = coverage->mImageChannel->getSettings();

    json["bounds"]["minLong"] = coverageSettings.mBounds.mMinLon;
    json["bounds"]["maxLong"] = coverageSettings.mBounds.mMaxLon;
    json["bounds"]["minLat"]  = coverageSettings.mBounds.mMinLat;
    json["bounds"]["maxLat"]  = coverageSettings.mBounds.mMaxLat;

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

void CoverageViewer::onMessageFromJS(nlohmann::json const& message) {
}

////////////////////////////////////////////////////////////////////////////////////////////////////

nlohmann::json CoverageViewer::getData() const {
  return nlohmann::json();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void CoverageViewer::setData(nlohmann::json const& json) {
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::visualquery
