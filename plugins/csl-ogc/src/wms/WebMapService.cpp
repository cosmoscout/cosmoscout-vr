////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "WebMapService.hpp"
#include "../logger.hpp"
#include "WebMapException.hpp"

#include "../../../src/cs-utils/filesystem.hpp"
#include "../../../src/cs-utils/utils.hpp"

#include <regex>

#include <boost/filesystem.hpp>

#include <curlpp/Easy.hpp>
#include <curlpp/Options.hpp>

namespace csl::ogc {

////////////////////////////////////////////////////////////////////////////////////////////////////

WebMapService::WebMapService(std::string url, CacheMode cacheMode, std::string cacheDir)
    : WebServiceBase(
          std::move(url), cacheMode, std::move(cacheDir), "WMS", "1.3.0", {"WMS_Capabilities"})
    , mSettings(parseSettings())
    , mMapFormats(parseMapFormats())
    , mRootLayer(parseRootLayer()) {
  setTitle(parseTitle());
  mRootLayer.getRequestableLayers(mRequestableLayers);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

WebMapService::Settings const& WebMapService::getSettings() const {
  return mSettings;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

WebMapLayer const& WebMapService::getRootLayer() const {
  return mRootLayer;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::vector<WebMapLayer> const& WebMapService::getLayers() const {
  return mRequestableLayers;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::optional<WebMapLayer> WebMapService::getLayer(std::string const& name) const {
  std::vector<WebMapLayer> layers = getLayers();
  auto                     layer  = std::find_if(
      layers.begin(), layers.end(), [name](WebMapLayer const& l) { return l.getName() == name; });
  if (layer == layers.end()) {
    return {};
  }

  return *layer;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool WebMapService::isFormatSupported(std::string const& format) const {
  return cs::utils::contains(mMapFormats, format);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::unique_ptr<OGCExceptionReport> WebMapService::createExceptionReport(
    VistaXML::TiXmlDocument const& doc) const {
  return std::make_unique<WebMapExceptionReport>(doc);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

WebMapLayer WebMapService::parseRootLayer() {
  VistaXML::TiXmlHandle   capabilityHandle(getCapabilities());
  VistaXML::TiXmlElement* root =
      capabilityHandle.FirstChildElement("Capability").FirstChildElement("Layer").ToElement();

  WebMapLayer::Settings settings;

  // Set default attribution to contact person if given
  VistaXML::TiXmlElement* contactPerson = capabilityHandle.FirstChildElement("Service")
                                              .FirstChildElement("ContactInformation")
                                              .FirstChildElement("ContactPersonPrimary")
                                              .ToElement();
  if (contactPerson != nullptr) {
    std::optional<std::string> organization =
        utils::getElementValue<std::string>(contactPerson, {"ContactOrganization"});
    std::optional<std::string> person =
        utils::getElementValue<std::string>(contactPerson, {"ContactPerson"});
    if (person.has_value() && !organization.has_value()) {
      settings.mAttribution = person.value();
    } else if (organization.has_value()) {
      settings.mAttribution = organization.value();
    }
  }

  return WebMapLayer(root, settings);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::string WebMapService::parseTitle() {
  return utils::getElementValue<std::string>(getCapabilities(), {"Service", "Title"})
      .value_or("Untitled");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

WebMapService::Settings WebMapService::parseSettings() {
  VistaXML::TiXmlHandle   capabilityHandle(getCapabilities());
  WebMapService::Settings settings;

  settings.mMaxWidth =
      utils::getElementValue<int>(capabilityHandle.ToElement(), {"Service", "MaxWidth"});
  settings.mMaxHeight =
      utils::getElementValue<int>(capabilityHandle.ToElement(), {"Service", "MaxHeight"});

  return settings;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::vector<std::string> WebMapService::parseMapFormats() {
  VistaXML::TiXmlHandle   capabilityHandle(getCapabilities());
  VistaXML::TiXmlElement* getMapCapability = capabilityHandle.FirstChildElement("Capability")
                                                 .FirstChildElement("Request")
                                                 .FirstChildElement("GetMap")
                                                 .ToElement();

  if (getMapCapability == nullptr) {
    logger().warn("Could not determine available file formats for '{}'.", getUrl());
    throw std::runtime_error("Capabilities parsing failed");
  }

  std::vector<std::string> mapFormats;

  for (VistaXML::TiXmlElement* formatElement   = getMapCapability->FirstChildElement("Format");
       formatElement != nullptr; formatElement = formatElement->NextSiblingElement("Format")) {
    std::optional<std::string> format = utils::getElementValue<std::string>(formatElement);
    if (format.has_value()) {
      mapFormats.push_back(format.value());
    }
  }

  return mapFormats;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csl::ogc
