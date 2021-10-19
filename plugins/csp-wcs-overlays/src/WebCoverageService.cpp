////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2020 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "WebCoverageService.hpp"
#include "WebCoverageException.hpp"
#include "logger.hpp"

#include "../../../src/cs-utils/filesystem.hpp"
#include "../../../src/cs-utils/utils.hpp"

#include <regex>

#include <boost/filesystem.hpp>

#include <curlpp/Easy.hpp>
#include <curlpp/Options.hpp>

namespace csp::wcsoverlays {

////////////////////////////////////////////////////////////////////////////////////////////////////

WebCoverageService::WebCoverageService(std::string url, CacheMode cacheMode, std::string cacheDir)
    : mUrl(std::move(url))
    , mCacheMode(cacheMode)
    , mCacheDir(std::move(cacheDir))
    , mCacheFileName(std::regex_replace(mUrl, std::regex("[/:*]"), "_") + ".xml")
    , mTitle(parseTitle()) {
  parseCoverages();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::string const& WebCoverageService::getUrl() const {
  return mUrl;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::string const& WebCoverageService::getTitle() const {
  return mTitle;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::vector<WebCoverage> const& WebCoverageService::getCoverages() const {
  return mRequestableCoverages;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::optional<WebCoverage> WebCoverageService::getCoverage(std::string const& title) const {
  std::vector<WebCoverage> coverages = getCoverages();

  auto coverage = std::find_if(
      coverages.begin(), coverages.end(), [title](WebCoverage const& capabilityCoverage) {
        return capabilityCoverage.getTitle() == title || capabilityCoverage.getId() == title;
      });

  if (coverage == coverages.end()) {
    return {};
  }

  return *coverage;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::optional<VistaXML::TiXmlDocument> WebCoverageService::getCapabilitiesFromCache() {
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

    VistaXML::TiXmlElement* root = cacheDoc.FirstChildElement("wcs:Capabilities");
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
WebCoverageService::checkUpdateSequence(VistaXML::TiXmlDocument cacheDoc) {

  // Get the update sequence number from the cached file, to check if it is up to date
  VistaXML::TiXmlElement* root           = cacheDoc.FirstChildElement("wcs:Capabilities");
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
      logger().warn("Failed to perform WCS Capabilities request while checking cache validity "
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
      WebCoverageExceptionReport e(resDoc);
      if (e.getExceptions().size() == 1 &&
          e.getExceptions()[0].getCode() == WebCoverageException::Code::eCurrentUpdateSequence) {
        // Cache is up to date
        return std::make_tuple(cacheDoc, std::nullopt);
      }
      logger().warn(
          "WCS Exception occurred while checking cache validity for '{}': '{}'!", mUrl, e.what());
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

VistaXML::TiXmlElement* WebCoverageService::getCapabilities() {
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

    VistaXML::TiXmlElement* capabilities = mDoc->FirstChildElement("wcs:Capabilities");

    if (capabilities == nullptr) {
      std::stringstream message;
      message << "WCS capabilities document for '" << mUrl << "' is not valid";
      throw std::runtime_error(message.str());
    }

    std::optional<std::string> version = utils::getAttribute<std::string>(capabilities, "version");

    if (!version.has_value()) {
      logger().warn("No version number given in capabilities! Trying to use server anyway.");
    } else {
      if (version.value() != "2.0.1") {
        std::stringstream message;
        message << "WCS '" << mUrl << "' only supports WCS version '" << version.value() << "'";
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

  VistaXML::TiXmlElement* capabilities = mDoc->FirstChildElement("wcs:Capabilities");
  return capabilities;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::tuple<VistaXML::TiXmlDocument, std::string> WebCoverageService::requestCapabilities() {
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
    message << "WCS capabilities request failed for '" << mUrl << "': '" << e.what() << "'";
    throw std::runtime_error(message.str());
  }

  std::string             docString = xmlStream.str();
  VistaXML::TiXmlDocument doc;
  doc.Parse(docString.c_str());
  if (doc.Error()) {
    std::stringstream message;
    message << "Parsing WCS capabilities failed for '" << mUrl << "': '" << doc.ErrorDesc() << "'";
    throw std::runtime_error(message.str());
  }

  // Check for WCS exception
  try {
    WebCoverageExceptionReport e(doc);
    std::stringstream          message;
    message << "Requesting WCS capabilities for '" << mUrl << "' resulted in WCS exception: '"
            << e.what() << "'";
    throw std::runtime_error(message.str());
  } catch (std::exception const&) {
    // No WCS exception occurred
  }
  return std::make_tuple(doc, docString);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void WebCoverageService::parseCoverages() {
  VistaXML::TiXmlHandle   capabilityHandle(getCapabilities());
  VistaXML::TiXmlElement* contents = capabilityHandle.FirstChildElement("wcs:Contents").ToElement();

  WebCoverage::Settings settings;

  // Set default attribution to contact person if given
  VistaXML::TiXmlElement* contactPerson =
      capabilityHandle.FirstChildElement("ows:ServiceProvider").ToElement();

  if (contactPerson != nullptr) {
    std::optional<std::string> organization =
        utils::getElementValue<std::string>(contactPerson, {"ows:ProviderName"});
    if (organization.has_value()) {
      settings.mAttribution = organization.value();
    }
  }

  for (auto* coverage = contents->FirstChild("wcs:CoverageSummary"); coverage;
       coverage       = coverage->NextSibling()) {
    if (!coverage->NoChildren()) {
      try {
        mRequestableCoverages.emplace_back(WebCoverage(coverage->ToElement(), settings, mUrl));
      } catch (std::exception const&) {
        // Coverage has no ID
      }
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::string WebCoverageService::parseTitle() {
  return utils::getElementValue<std::string>(
      getCapabilities(), {"ows:ServiceIdentification", "ows:Title"})
      .value_or("Untitled");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::stringstream WebCoverageService::getGetCapabilitiesUrl() const {
  std::stringstream urlStream;
  urlStream << mUrl;
  urlStream << "?SERVICE=WCS";
  urlStream << "&VERSION=2.0.1";
  urlStream << "&REQUEST=GetCapabilities";

  return urlStream;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::wcsoverlays