////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2020 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "WebMapService.hpp"
#include "WebMapException.hpp"
#include "logger.hpp"

#include "../../../src/cs-utils/filesystem.hpp"
#include "../../../src/cs-utils/utils.hpp"

#include <regex>

#include <boost/filesystem.hpp>

#include <curlpp/Easy.hpp>
#include <curlpp/Options.hpp>

namespace csp::wmsoverlays {

////////////////////////////////////////////////////////////////////////////////////////////////////

WebMapService::WebMapService(std::string url, CacheMode cacheMode, std::string cacheDir)
    : mUrl(std::move(url))
    , mCacheMode(cacheMode)
    , mCacheDir(std::move(cacheDir))
    , mCacheFileName(std::regex_replace(mUrl, std::regex("[/:*]"), "_") + ".xml")
    , mTitle(parseTitle())
    , mSettings(parseSettings())
    , mMapFormats(parseMapFormats())
    , mRootLayer(parseRootLayer()) {
  mRootLayer.getRequestableLayers(mRequestableLayers);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::string const& WebMapService::getUrl() const {
  return mUrl;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::string const& WebMapService::getTitle() const {
  return mTitle;
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

VistaXML::TiXmlElement* WebMapService::getCapabilities() {
  if (!mDoc.has_value()) {
    std::optional<std::string> docString;
    bool                       saveToCache = false;

    // Check cache for capability document according to cache mode
    switch (mCacheMode) {
    case CacheMode::eAlways: {
      saveToCache   = true;
      auto cacheDoc = getCapabilitiesFromCache();
      if (cacheDoc.has_value()) {
        mDoc = cacheDoc.value();
      }
      break;
    }
    case CacheMode::eUpdateSequence: {
      saveToCache   = true;
      auto cacheDoc = getCapabilitiesFromCache();
      if (cacheDoc.has_value()) {
        auto checkRes = checkUpdateSequence(cacheDoc.value());
        if (checkRes.has_value()) {
          std::tie(mDoc, docString) = checkRes.value();
        }
      }
      break;
    }
    case CacheMode::eNever: {
      saveToCache = false;
      break;
    }
    }

    if (!mDoc.has_value()) {
      // No valid data found in cache, request capabilities from server
      std::tie(mDoc, docString) = requestCapabilities();
    }

    VistaXML::TiXmlElement* capabilities = mDoc->FirstChildElement("WMS_Capabilities");
    if (capabilities == nullptr) {
      std::stringstream message;
      message << "WMS capabilities document for '" << mUrl << "' is not valid";
      throw std::runtime_error(message.str());
    }
    std::optional<std::string> version = utils::getAttribute<std::string>(capabilities, "version");
    if (!version.has_value()) {
      logger().warn("No version number given in capabilities! Trying to use server anyway.");
    } else {
      if (version.value() != "1.3.0") {
        std::stringstream message;
        message << "WMS '" << mUrl << "' only supports WMS version '" << version.value() << "'";
        throw std::runtime_error(message.str());
      }
    }

    if (saveToCache && docString.has_value()) {
      // Save capabilities to cache
      boost::filesystem::path cacheFile(mCacheFileName);
      boost::filesystem::path cacheDir(mCacheDir);
      boost::filesystem::path cacheFilePath(cacheDir / cacheFile);

      auto cacheDirAbs(boost::filesystem::absolute(cacheDir));
      if (!(boost::filesystem::exists(cacheDirAbs))) {
        try {
          cs::utils::filesystem::createDirectoryRecursively(cacheDirAbs);
        } catch (std::exception& e) {
          logger().warn("Failed to create cache directory: {}!", e.what());
        }
      }
      cs::utils::filesystem::writeStringToFile(cacheFilePath.string(), docString.value());
    }
  }
  VistaXML::TiXmlElement* capabilities = mDoc->FirstChildElement("WMS_Capabilities");
  return capabilities;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::optional<VistaXML::TiXmlDocument> WebMapService::getCapabilitiesFromCache() {
  boost::filesystem::path cacheFile(mCacheFileName);
  boost::filesystem::path cacheDir(mCacheDir);
  boost::filesystem::path cacheFilePath(cacheDir / cacheFile);

  // Check if file with the correct name is in the cache
  if (boost::filesystem::exists(cacheFilePath) && boost::filesystem::file_size(cacheFilePath) > 0) {
    std::string capabilitiesString = cs::utils::filesystem::loadToString(cacheFilePath.string());

    VistaXML::TiXmlDocument cacheDoc;
    cacheDoc.Parse(capabilitiesString.c_str());
    if (cacheDoc.Error()) {
      logger().warn("Failed to parse cached file for '{}': '{}'!", mUrl, cacheDoc.ErrorDesc());
      return {};
    }

    VistaXML::TiXmlElement* root = cacheDoc.FirstChildElement("WMS_Capabilities");
    if (root == nullptr) {
      logger().warn(
          "Cached capabilities document for '{}' is not valid! Requesting a new one.", mUrl);
      return {};
    }
    return cacheDoc;
  }
  // No file found in cache
  return {};
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::optional<std::tuple<VistaXML::TiXmlDocument, std::optional<std::string>>>
WebMapService::checkUpdateSequence(VistaXML::TiXmlDocument cacheDoc) {
  // This method uses the updateSequence value in the cached capabilities according to 7.2.3.5 of
  // the WMS 1.3.0 implementation specification to check if a new capability document should be
  // requested.

  // Get the update sequence number from the cached file, to check if it is up to date
  VistaXML::TiXmlElement* root           = cacheDoc.FirstChildElement("WMS_Capabilities");
  const char*             updateSequence = root->Attribute("updateSequence");
  if (updateSequence != nullptr) {
    // A sequence number was found, now check if it is the most recent one
    std::stringstream url = getGetCapabilitiesUrl();
    url << "&UPDATESEQUENCE=" << updateSequence;

    std::stringstream resStream;
    curlpp::Easy      request;
    request.setOpt(curlpp::options::Url(url.str()));
    request.setOpt(curlpp::options::WriteStream(&resStream));
    request.setOpt(curlpp::options::NoSignal(true));
    request.setOpt(curlpp::options::SslVerifyPeer(false));

    try {
      request.perform();
    } catch (std::exception const& e) {
      logger().warn("Failed to perform WMS Capabilities request while checking cache validity "
                    "for '{}': '{}'!",
          mUrl, e.what());
      return {};
    }

    const std::string       resString = resStream.str();
    VistaXML::TiXmlDocument resDoc;
    resDoc.Parse(resString.c_str());
    if (resDoc.Error()) {
      logger().warn("Parsing XML failed while checking cache validity for '{}': '{}'!", mUrl,
          resDoc.ErrorDesc());
      return {};
    }

    try {
      WebMapExceptionReport e(resDoc);
      if (e.getExceptions().size() == 1 &&
          e.getExceptions()[0].getCode() == WebMapException::Code::eCurrentUpdateSequence) {
        // Cache is up to date
        return std::make_tuple(cacheDoc, std::nullopt);
      }
      logger().warn(
          "WMS Exception occurred while checking cache validity for '{}': '{}'!", mUrl, e.what());
      // Cache is not up to date, and an exception occurred
      return {};
    } catch (std::exception const&) {
      // No exception, the request's result should be the newest capabilities
      return std::make_tuple(resDoc, resString);
    }
  } else {
    // No sequence number found, so we can't verify that our cached file is up to date
    return {};
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::tuple<VistaXML::TiXmlDocument, std::string> WebMapService::requestCapabilities() {
  std::stringstream url = getGetCapabilitiesUrl();

  std::stringstream xmlStream;
  curlpp::Easy      request;
  request.setOpt(curlpp::options::Url(url.str()));
  request.setOpt(curlpp::options::WriteStream(&xmlStream));
  request.setOpt(curlpp::options::NoSignal(true));
  request.setOpt(curlpp::options::SslVerifyPeer(false));

  try {
    request.perform();
  } catch (std::exception const& e) {
    std::stringstream message;
    message << "WMS capabilities request failed for '" << mUrl << "': '" << e.what() << "'";
    throw std::runtime_error(message.str());
  }

  std::string             docString = xmlStream.str();
  VistaXML::TiXmlDocument doc;
  doc.Parse(docString.c_str());
  if (doc.Error()) {
    std::stringstream message;
    message << "Parsing WMS capabilities failed for '" << mUrl << "': '" << doc.ErrorDesc() << "'";
    throw std::runtime_error(message.str());
  }

  // Check for WMS exception
  try {
    WebMapExceptionReport e(doc);
    std::stringstream     message;
    message << "Requesting WMS capabilities for '" << mUrl << "' resulted in WMS exception: '"
            << e.what() << "'";
    throw std::runtime_error(message.str());
  } catch (std::exception const&) {
    // No WMS exception occurred
  }
  return std::make_tuple(doc, docString);
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
    logger().warn("Could not determine available file formats for '{}'.", mUrl);
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

std::stringstream WebMapService::getGetCapabilitiesUrl() const {
  std::stringstream urlStream;
  urlStream << mUrl;
  urlStream << "?SERVICE=WMS";
  urlStream << "&VERSION=1.3.0";
  urlStream << "&REQUEST=GetCapabilities";
  return urlStream;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::wmsoverlays
