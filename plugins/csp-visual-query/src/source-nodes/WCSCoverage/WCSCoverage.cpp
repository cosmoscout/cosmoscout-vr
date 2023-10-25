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

std::unique_ptr<WCSCoverage> WCSCoverage::sCreate(
  std::shared_ptr<std::vector<csl::ogc::WebCoverageService>> wcs) {
  return std::make_unique<WCSCoverage>(std::move(wcs));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

WCSCoverage::WCSCoverage(std::shared_ptr<std::vector<csl::ogc::WebCoverageService>> wcs)
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

  // Send available servers
  if (message.dump() == "\"requestServers\"") {
    sendServersToJs();
    return;
  }

  // set newly selected server
  if (message.find("server") != message.end()) {

    // reset server and image channel selection
    if (message["server"] == "none") {
      mSelectedServer = nullptr;
      mSelectedImageChannel = nullptr;

      nlohmann::json imageChannel;
      imageChannel["imageChannel"] = "reset";
      sendMessageToJS(imageChannel);

    // set new server and send available image channels
    } else {
      for (auto wcs : (*mWcs)) {
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
  else if (message.find("imageChannel") !=  message.end()) {

    // reset image channel selection
    if (message["imageChannel"] == "none") {
      mSelectedImageChannel = nullptr;

    // set new image channel
    } else {
      auto temp = mSelectedServer->getCoverage(message["imageChannel"]);
      mSelectedImageChannel = (temp.has_value() ? std::make_shared<csl::ogc::WebCoverage>(temp.value()) : nullptr);
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
  if (mSelectedServer != nullptr) {
    data["serverUrl"] = mSelectedServer->getUrl();

    if (mSelectedImageChannel != nullptr) {
      data["imageChannelId"] = mSelectedImageChannel->getId();
    }
  }
  return data;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void WCSCoverage::setData(nlohmann::json const& json) {
  if (json.find("serverUrl") != json.end()) {

    for (csl::ogc::WebCoverageService wcs : *mWcs) {
      if (wcs.getUrl() == json["serverUrl"]) {
        mSelectedServer = std::make_shared<csl::ogc::WebCoverageService>(wcs);
        break;
      }
    }

    if (mSelectedServer != nullptr && json.find("imageChannelId") != json.end()) {

      auto temp = mSelectedServer->getCoverage(json["imageChannelId"]);
      mSelectedImageChannel = (temp.has_value() ? std::make_shared<csl::ogc::WebCoverage>(temp.value()) : nullptr);
    }
  }
  // process() ?
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void WCSCoverage::process() {
  if (mSelectedServer == nullptr || mSelectedImageChannel == nullptr) {
    return;
  }

  auto coverageSettings = mSelectedImageChannel->getSettings();

  // writeOutput("minTimeValue", coverageSettings); ???
  // writeOutput("maxTimeValue", coverageSettings); ???
  writeOutput("lngBoundMinOut", coverageSettings.mBounds.mMinLon);
  writeOutput("lngBoundMaxOut", coverageSettings.mBounds.mMaxLon);
  writeOutput("latBoundMinOut", coverageSettings.mBounds.mMinLat);
  writeOutput("latBoundMaxOut", coverageSettings.mBounds.mMaxLat);

  writeOutput("coverageOut", std::make_shared<CoverageContainer>(mSelectedServer, mSelectedImageChannel));
}

void WCSCoverage::sendServersToJs() {
  std::vector<std::string> serverNames;
  for (csl::ogc::WebCoverageService server : *mWcs) {
    serverNames.push_back(server.getTitle());
  }
  nlohmann::json server;
  server["server"] = serverNames;
  sendMessageToJS(server);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void WCSCoverage::sendImageChannelsToJs() {
  std::vector<std::string> imageChannelNames;
  
  for (csl::ogc::WebCoverage imageChannel : mSelectedServer->getCoverages()) {
    imageChannelNames.push_back(imageChannel.getTitle());
  }
  nlohmann::json imageChannels;
  imageChannels["imageChannel"] = imageChannelNames;
  sendMessageToJS(imageChannels);
}

} // namespace csp::visualquery
