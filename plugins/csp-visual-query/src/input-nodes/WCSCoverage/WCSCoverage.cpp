////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "WCSCoverage.hpp"

#include "../../../../../src/cs-utils/filesystem.hpp"
#include "../../../../csl-ogc/src/wcs/WebCoverageService.hpp"
#include "../../../../csl-ogc/src/wcs/WebCoverageTextureLoader.hpp"
#include "../../logger.hpp"
#include "../../types/CoverageContainer.hpp"
#include "../../types/types.hpp"

namespace csp::visualquery {

////////////////////////////////////////////////////////////////////////////////////////////////////

const std::string WCSCoverage::sName = "WCSCoverage";

////////////////////////////////////////////////////////////////////////////////////////////////////

std::string WCSCoverage::sSource() {
  return cs::utils::filesystem::loadToString(
      "../share/resources/nodes/csp-visual-query/WCSCoverage.js");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::unique_ptr<WCSCoverage> WCSCoverage::sCreate(std::vector<csl::ogc::WebCoverageService> wcs) {
  return std::make_unique<WCSCoverage>(std::move(wcs));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

WCSCoverage::WCSCoverage(std::vector<csl::ogc::WebCoverageService> wcs)
    : mWcs(std::move(wcs))
    , mSelectedServer(nullptr)
    , mSelectedImageChannel(nullptr) {
}

////////////////////////////////////////////////////////////////////////////////////////////////////

WCSCoverage::~WCSCoverage() {
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::string const& WCSCoverage::getName() const {
  return sName;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void WCSCoverage::onMessageFromJS(nlohmann::json const& message) {

  logger().debug("WCSCoverage: Message form JS: {}", message.dump());

  // set newly selected server
  if (message.find("server") != message.end()) {

    // reset server and image channel selection
    if (message["server"] == "None") {
      mSelectedServer       = nullptr;
      mSelectedImageChannel = nullptr;

      nlohmann::json imageChannel;
      imageChannel["imageChannels"] = "reset";
      sendMessageToJS(imageChannel);

      // set new server and send available image channels
    } else {
      for (auto const& wcs : mWcs) {
        if (wcs.getTitle() == message["server"]) {
          mSelectedServer = std::make_shared<csl::ogc::WebCoverageService>(wcs);
          break;
        }
      }
      sendImageChannelsToJs();
    }
    return;
  }

  // set newly selected image channel
  if (message.find("imageChannel") != message.end()) {

    // reset image channel selection
    if (message["imageChannel"] == "None") {
      mSelectedImageChannel = nullptr;

      // set new image channel
    } else {
      auto temp = mSelectedServer->getCoverage(message["imageChannel"]);
      mSelectedImageChannel =
          (temp.has_value() ? std::make_shared<csl::ogc::WebCoverage>(temp.value()) : nullptr);
    }
  }

  // Whenever the server changes, we write the new output by calling the process() method.
  // Writing the output will not trigger a graph reprocessing right away, it will only queue
  // up the connected nodes for being processed in the next update step.
  process();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

nlohmann::json WCSCoverage::getData() const {
  nlohmann::json data;

  std::vector<std::string> serverNames{"None"};
  for (csl::ogc::WebCoverageService const& server : mWcs) {
    serverNames.push_back(server.getTitle());
  }

  data["servers"] = serverNames;

  if (mSelectedServer != nullptr) {
    data["selectedURL"]    = mSelectedServer->getUrl();
    data["selectedServer"] = mSelectedServer->getTitle();

    std::vector<std::string> imageChannelNames{"None"};

    for (csl::ogc::WebCoverage const& imageChannel : mSelectedServer->getCoverages()) {
      imageChannelNames.push_back(imageChannel.getTitle());
    }
    data["coverages"] = imageChannelNames;

    if (mSelectedImageChannel != nullptr) {
      data["selectedCoverageId"] = mSelectedImageChannel->getId();
      data["selectedCoverage"]   = mSelectedImageChannel->getTitle();
    }
  }
  return data;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void WCSCoverage::setData(nlohmann::json const& json) {
  if (json.find("selectedURL") != json.end()) {

    for (csl::ogc::WebCoverageService const& wcs : mWcs) {
      if (wcs.getUrl() == json["selectedURL"]) {
        mSelectedServer = std::make_shared<csl::ogc::WebCoverageService>(wcs);
        break;
      }
    }

    if (mSelectedServer != nullptr && json.find("selectedCoverage") != json.end()) {

      auto temp = mSelectedServer->getCoverage(json["selectedCoverageId"]);
      mSelectedImageChannel =
          (temp.has_value() ? std::make_shared<csl::ogc::WebCoverage>(temp.value()) : nullptr);
    }
  }
  // process() ?
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void WCSCoverage::init() {
  std::vector<std::string> serverNames{"None"};
  for (csl::ogc::WebCoverageService const& server : mWcs) {
    serverNames.push_back(server.getTitle());
  }
  nlohmann::json server;
  server["servers"] = serverNames;
  sendMessageToJS(server);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void WCSCoverage::process() {
  if (mSelectedServer == nullptr || mSelectedImageChannel == nullptr) {
    return;
  }

  mSelectedImageChannel->update();
  auto coverageSettings = mSelectedImageChannel->getSettings();

  std::array<double, 4> bounds{coverageSettings.mBounds.mMinLon, coverageSettings.mBounds.mMaxLon,
      coverageSettings.mBounds.mMinLat, coverageSettings.mBounds.mMaxLat};

  writeOutput("timeIntervalsOut", coverageSettings.mTimeIntervals);
  writeOutput("boundsOut", bounds);
  writeOutput(
      "coverageOut", std::make_shared<CoverageContainer>(mSelectedServer, mSelectedImageChannel));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void WCSCoverage::sendImageChannelsToJs() {
  std::vector<std::string> imageChannelNames{"None"};

  for (csl::ogc::WebCoverage const& imageChannel : mSelectedServer->getCoverages()) {
    imageChannelNames.push_back(imageChannel.getTitle());
  }
  nlohmann::json imageChannels;
  imageChannels["imageChannels"] = imageChannelNames;
  sendMessageToJS(imageChannels);
}

} // namespace csp::visualquery