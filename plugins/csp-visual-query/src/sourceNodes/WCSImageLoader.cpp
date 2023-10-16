////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "WCSImageLoader.hpp"

#include "../../../../src/cs-utils/filesystem.hpp"
#include "../../../csl-ogc/src/wcs/WebCoverageService.hpp"
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
  : mWcs(std::move(wcs)) {
  
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
  
  if (message.dump() == "\"requestServers\"") {
    sendServersToJs();
  
  } else {
    mSelectedWcsIndex = message; 
    // Send image layers for newly selected server
    sendImageLayersToJs();
  }

  // Whenever the server changes, we write the new output by calling the process() method.
  // Writing the output will not trigger a graph reprocessing right away, it will only queue
  // up the connected nodes for being processed in the next update step.
  process();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

nlohmann::json WCSImageLoader::getData() const {
  return {{"url", mSelectedWcsIndex}};
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void WCSImageLoader::setData(nlohmann::json const& json) {
  mSelectedWcsIndex = json["url"];
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void WCSImageLoader::process() {
  // double first  = readInput<double>("first", 0.0);

  // The name of the port must match the name given in the JavaScript code above.
  writeOutput("minDataValue", 0);
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

void WCSImageLoader::sendImageLayersToJs() {
  std::vector<std::string> imageLayerNames;
  
  for (csl::ogc::WebCoverage imageLayer : (*mWcs)[mSelectedWcsIndex].getCoverages()) {
    imageLayerNames.push_back(imageLayer.getTitle());
  }
  nlohmann::json imageLayers;
  imageLayers["imageLayer"] = imageLayerNames;
  sendMessageToJS(imageLayers);
}

} // namespace csp::visualquery
