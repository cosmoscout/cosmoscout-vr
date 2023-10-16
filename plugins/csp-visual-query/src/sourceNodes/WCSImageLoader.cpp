////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "WCSImageLoader.hpp"

#include "../../../../src/cs-utils/filesystem.hpp"
#include "../../../csl-ogc/src/wcs/WebCoverageService.hpp"
#include "../../../csl-ogc/src/wcs/WebCoverageTextureLoader.hpp"
#include "../types/types.hpp"

namespace csp::visualquery {

////////////////////////////////////////////////////////////////////////////////////////////////////

const std::string WCSImageLoader::sName = "WCSImageLoader";

////////////////////////////////////////////////////////////////////////////////////////////////////

std::string WCSImageLoader::sSource() {
  return cs::utils::filesystem::loadToString(
      "../share/resources/nodes/csp-visual-query/WCSImageLoader.js");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::unique_ptr<WCSImageLoader> WCSImageLoader::sCreate(
  std::shared_ptr<std::vector<csl::ogc::WebCoverageService>> wcs) {
  return std::make_unique<WCSImageLoader>(std::move(wcs));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

WCSImageLoader::WCSImageLoader(std::shared_ptr<std::vector<csl::ogc::WebCoverageService>> wcs)
  : mWcs(std::move(wcs))
  , mSelectedServer(nullptr)
  , mSelectedImageChannel(nullptr) {
}

////////////////////////////////////////////////////////////////////////////////////////////////////

WCSImageLoader::~WCSImageLoader() {
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::string const& WCSImageLoader::getName() const {
  return sName;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void WCSImageLoader::onMessageFromJS(nlohmann::json const& message) {

  logger().debug("Message form JS: {}", message.dump());

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

nlohmann::json WCSImageLoader::getData() const {
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

void WCSImageLoader::setData(nlohmann::json const& json) {
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

void WCSImageLoader::process() {
  // double first  = readInput<double>("first", 0.0);

  // The name of the port must match the name given in the JavaScript code above.
  // writeOutput("minDataValue", 0);

  // create request for texture loading
  if (mSelectedServer == nullptr || mSelectedImageChannel == nullptr) {
    return;
  }

  logger().debug("process!");

  csl::ogc::WebCoverageTextureLoader::Request request;
  
  request.mTime = std::to_string(readInput<double>("wcsTime", 0.0));
  
  logger().debug(0);

  csl::ogc::Bounds bound;
  bound.mMinLon = readInput<double>("xBoundMin", 0.0);
  bound.mMaxLon = readInput<double>("xBoundMax", 100.0);
  bound.mMinLat = readInput<double>("yBoundMin", 0.0);
  bound.mMaxLat = readInput<double>("yBoundMax", 100.0);
  request.mBounds = bound;

  logger().debug(1);

  request.mMaxSize = readInput<int>("resolution", 1024);

  logger().debug(2);

  request.mFormat = "image/tiff";

  logger().debug(3);

  auto texLoader = csl::ogc::WebCoverageTextureLoader();
  auto texture = texLoader.loadTexture(*mSelectedServer, *mSelectedImageChannel, request, 
    "../../../install/windows-Release/share/cache/csp-visual-query/texture-cache", true);

  logger().debug(4);

  if (texture.has_value()) {
    logger().debug("x in texture: {}", texture.value().x);
    logger().debug("y in texture: {}", texture.value().y);
  }

  logger().debug(5);
}

void WCSImageLoader::sendServersToJs() {
  std::vector<std::string> serverNames;
  for (csl::ogc::WebCoverageService server : *mWcs) {
    serverNames.push_back(server.getTitle());
  }
  nlohmann::json server;
  server["server"] = serverNames;
  sendMessageToJS(server);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void WCSImageLoader::sendImageChannelsToJs() {
  std::vector<std::string> imageChannelNames;
  
  for (csl::ogc::WebCoverage imageChannel : mSelectedServer->getCoverages()) {
    imageChannelNames.push_back(imageChannel.getTitle());
  }
  nlohmann::json imageChannels;
  imageChannels["imageChannel"] = imageChannelNames;
  sendMessageToJS(imageChannels);
}

} // namespace csp::visualquery
