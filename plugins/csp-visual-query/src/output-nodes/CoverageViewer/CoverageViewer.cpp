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

  if (coverage == nullptr) {
    nlohmann::json jsonBounds;
    jsonBounds["bounds"]["minLong"] = "";
    jsonBounds["bounds"]["maxLong"] = "";
    jsonBounds["bounds"]["minLat"]  = "";
    jsonBounds["bounds"]["maxLat"]  = "";
    sendMessageToJS(jsonBounds);
    return;
  }

  auto bounds = coverage->mImageChannel->getSettings().mBounds;

  nlohmann::json jsonBounds;
  jsonBounds["bounds"]["minLong"] = bounds.mMinLon;
  jsonBounds["bounds"]["maxLong"] = bounds.mMaxLon;
  jsonBounds["bounds"]["minLat"]  = bounds.mMinLat;
  jsonBounds["bounds"]["maxLat"]  = bounds.mMaxLat;
  sendMessageToJS(jsonBounds);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void CoverageViewer::onMessageFromJS(nlohmann::json const& message) {
  process();
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
