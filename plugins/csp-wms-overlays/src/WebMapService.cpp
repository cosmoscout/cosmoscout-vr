////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2020 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "WebMapService.hpp"
#include "logger.hpp"

#include "../../../src/cs-utils/filesystem.hpp"

#include <regex>

#include <boost/filesystem.hpp>

#include <curlpp/Easy.hpp>
#include <curlpp/Options.hpp>

namespace csp::wmsoverlays {

////////////////////////////////////////////////////////////////////////////////////////////////////

WebMapService::WebMapService(std::string url, std::string cacheDir)
    : mUrl(url)
    , mCacheFileName(std::regex_replace(mUrl, std::regex("[/:*]"), "_") + ".xml")
    , mCacheDir(cacheDir)
    , mTitle(parseTitle())
    , mRootLayer(parseRootLayer()) {
  mRootLayer.getRequestableLayers(mRequestableLayers);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::string WebMapService::getUrl() const {
  return mUrl;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::string WebMapService::getTitle() const {
  return mTitle;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::vector<WebMapLayer> WebMapService::getLayers() const {
  return mRequestableLayers;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::optional<WebMapLayer> WebMapService::getLayer(std::string name) const {
  std::vector<WebMapLayer> layers = getLayers();
  auto                     layer  = std::find_if(
      layers.begin(), layers.end(), [&name](WebMapLayer l) { return l.getName() == name; });
  if (layer == layers.end()) {
    return {};
  } else {
    return *layer;
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

VistaXML::TiXmlElement* WebMapService::getCapabilities() {
  if (!mDoc.has_value()) {
    VistaXML::TiXmlDocument doc;
    std::string             docString;

    auto cacheRes = getCapabilitiesFromCache();

    if (cacheRes.has_value()) {
      std::tie(docString, doc) = cacheRes.value();
    } else {
      // No valid data found in cache, request capabilities from server
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
        logger().warn("Failed to perform WMS Capabilities request: '{}'! Exception: '{}'",
            url.str(), e.what());
        throw std::exception("Capabilities request failed");
      }

      docString = xmlStream.str();
      doc.Parse(docString.c_str());
      if (doc.Error()) {
        logger().warn("Parsing failed with '{}'", doc.ErrorDesc());
        throw std::exception("Capabilities parsing failed");
      }
    }

    // Cache file
    boost::filesystem::path cacheFile(mCacheFileName);
    boost::filesystem::path cacheDir(mCacheDir);
    boost::filesystem::path cacheFilePath(cacheDir / cacheFile);

    auto cacheDirAbs(boost::filesystem::absolute(cacheDir));
    if (!(boost::filesystem::exists(cacheDirAbs))) {
      try {
        cs::utils::filesystem::createDirectoryRecursively(cacheDirAbs);
      } catch (std::exception& e) {
        logger().error("Failed to create cache directory: {}", e.what());
      }
    }

    cs::utils::filesystem::writeStringToFile(cacheFilePath.string(), docString);

    mDoc = doc;
  }
  return mDoc->FirstChildElement("WMS_Capabilities");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::optional<std::pair<std::string, VistaXML::TiXmlDocument>>
WebMapService::getCapabilitiesFromCache() {
  boost::filesystem::path cacheFile(mCacheFileName);
  boost::filesystem::path cacheDir(mCacheDir);
  boost::filesystem::path cacheFilePath(cacheDir / cacheFile);

  // Check if file with the correct name is in the cache
  if (boost::filesystem::exists(cacheFilePath) && boost::filesystem::file_size(cacheFilePath) > 0) {
    std::string capabilitiesString = cs::utils::filesystem::loadToString(cacheFilePath.string());

    VistaXML::TiXmlDocument cacheDoc;
    cacheDoc.Parse(capabilitiesString.c_str());
    if (cacheDoc.Error()) {
      logger().warn("Failed to parse cached file: '{}'", cacheDoc.ErrorDesc());
      return {};
    }

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
        logger().warn("Failed to perform WMS Capabilities request for cache verification: '{}'! "
                      "Exception: '{}'",
            url.str(), e.what());
        return {};
      }

      const std::string       resString = resStream.str();
      VistaXML::TiXmlDocument resDoc;
      resDoc.Parse(resString.c_str());
      if (resDoc.Error()) {
        logger().trace("Parsing failed with '{}'", resDoc.ErrorDesc());
        return {};
      }

      VistaXML::TiXmlHandle   resRoot   = resDoc.FirstChildElement("ServiceExceptionReport");
      VistaXML::TiXmlElement* exception = resRoot.FirstChildElement("ServiceException").ToElement();

      if (exception == nullptr) {
        // No exception, the result should be the newest capabilities
        return std::make_pair(resString, resDoc);
      }
      const char* exceptionCode = exception->Attribute("code");

      if (exceptionCode != nullptr &&
          std::string(exceptionCode) == std::string("CurrentUpdateSequence")) {
        // Cache is up to date
        return std::make_pair(capabilitiesString, cacheDoc);
      } else {
        // Cache is not up to date, and an exception occured
        return {};
      }
    } else {
      // No sequence number found, so we can't verify that our cached file is up to date
      return {};
    }
  }
  // No file found in cache
  return {};
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
    std::stringstream    contact;
    VistaXML::TiXmlNode* organization =
        contactPerson->FirstChildElement("ContactOrganization")->FirstChild();
    VistaXML::TiXmlNode* person = contactPerson->FirstChildElement("ContactPerson")->FirstChild();
    if (person != nullptr) {
      contact << person->ValueStr();
    }
    if (person != nullptr && organization != nullptr) {
      contact << ", ";
    }
    if (organization != nullptr) {
      contact << organization->ValueStr();
    }
    settings.mAttribution = contact.str();
  }

  return WebMapLayer(root, settings);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::string WebMapService::parseTitle() {
  VistaXML::TiXmlHandle capabilityHandle(getCapabilities());
  VistaXML::TiXmlText*  serviceTitle = capabilityHandle.FirstChildElement("Service")
                                          .FirstChildElement("Title")
                                          .FirstChild()
                                          .ToText();
  return serviceTitle->ValueStr();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::stringstream WebMapService::getGetCapabilitiesUrl() {
  std::stringstream urlStream;
  urlStream << mUrl;
  urlStream << "?SERVICE=WMS";
  urlStream << "&VERSION=1.3.0";
  urlStream << "&REQUEST=GetCapabilities";
  return urlStream;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::wmsoverlays
