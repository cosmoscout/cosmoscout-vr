////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "WebServiceBase.hpp"

#include "../../../../src/cs-utils/filesystem.hpp"
#include "../wms/WebMapException.hpp"
#include "curlpp/Easy.hpp"
#include "curlpp/Options.hpp"
#include <boost/filesystem.hpp>
#include <utility>

namespace csl::ogc {

////////////////////////////////////////////////////////////////////////////////////////////////////

WebServiceBase::WebServiceBase(std::string url, CacheMode cacheMode, std::string cacheDir,
    std::string title, std::string serviceType, std::string supportedVersion, TagNames tagNames)
    : mUrl(std::move(url))
    , mCacheMode(cacheMode)
    , mCacheDir(std::move(cacheDir))
    , mCacheFileName(std::regex_replace(mUrl, std::regex("[/:*]"), "_") + ".xml")
    , mTitle(std::move(title))
    , mServiceType(std::move(serviceType))
    , mSupportedVersion(std::move(supportedVersion))
    , mTagNames(std::move(tagNames)) {
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::string const& WebServiceBase::getUrl() const noexcept {
  return mUrl;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::string const& WebServiceBase::getTitle() const noexcept {
  return mTitle;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

VistaXML::TiXmlElement* WebServiceBase::getCapabilities() {
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

    VistaXML::TiXmlElement* capabilities = mDoc->FirstChildElement(mTagNames.mCapabilitiesRoot);

    if (capabilities == nullptr) {
      throw std::runtime_error(
          fmt::format("{} capabilities document for '{}' is not valid!", mServiceType, mUrl));
    }

    std::optional<std::string> version = utils::getAttribute<std::string>(capabilities, "version");

    if (!version.has_value()) {
      logger().warn("No version number given in capabilities! Trying to use server anyway.");
    } else if (version.value() != mSupportedVersion) {
      throw std::runtime_error(fmt::format("{} '{}' only supports {} version '{}'!", mServiceType,
          mUrl, mServiceType, version.value()));
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

  VistaXML::TiXmlElement* capabilities = mDoc->FirstChildElement(mTagNames.mCapabilitiesRoot);
  return capabilities;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::optional<VistaXML::TiXmlDocument> WebServiceBase::getCapabilitiesFromCache() {
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

    VistaXML::TiXmlElement* root = cacheDoc.FirstChildElement(mTagNames.mCapabilitiesRoot);
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

std::tuple<VistaXML::TiXmlDocument, std::string> WebServiceBase::requestCapabilities() {
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
    throw std::runtime_error(
        fmt::format("OGC capabilities request failed for '{}': '{}'", mUrl, e.what()));
  }

  std::string             docString = xmlStream.str();
  VistaXML::TiXmlDocument doc;
  doc.Parse(docString.c_str());
  if (doc.Error()) {
    throw std::runtime_error(
        fmt::format("Parsing WCS capabilities failed for '{}': '{}'", mUrl, doc.ErrorDesc()));
  }

  std::unique_ptr<OGCExceptionReport> exceptionReport = createExceptionReport(doc);

  if (!exceptionReport->getExceptions().empty()) {
    throw std::runtime_error(
        fmt::format("Requesting OGC capabilities for '{}' resulted in OGC exception: '{}'", mUrl,
            exceptionReport->what()));
  }

  return std::make_tuple(doc, docString);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::optional<std::tuple<VistaXML::TiXmlDocument, std::optional<std::string>>>
WebServiceBase::checkUpdateSequence(VistaXML::TiXmlDocument cacheDoc) {
  // Get the update sequence number from the cached file, to check if it is up-to-date
  VistaXML::TiXmlElement* root           = cacheDoc.FirstChildElement(mTagNames.mCapabilitiesRoot);
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
      logger().warn("Failed to perform {} Capabilities request while checking cache validity "
                    "for '{}': '{}'!",
          mServiceType, mUrl, e.what());
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

    std::unique_ptr<OGCExceptionReport> exceptionReport = createExceptionReport(resDoc);

    if (exceptionReport->getExceptions().empty()) {
      return std::make_tuple(resDoc, resString);
    }

    if (exceptionReport->getExceptions().size() == 1 &&
        exceptionReport->getExceptions()[0]->getCode() == "CurrentUpdateSequence") {
      // Cache is up-to-date
      return std::make_tuple(cacheDoc, std::nullopt);
    }

    logger().warn("{} Exception occurred while checking cache validity for '{}': '{}'!",
        mServiceType, mUrl, exceptionReport->what());

    // Cache is not up-to-date, and an exception occurred
    return {};
  }

  // No sequence number found, so we can't verify that our cached file is up-to-date
  return {};
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::stringstream WebServiceBase::getGetCapabilitiesUrl() const noexcept {
  std::stringstream urlStream;
  urlStream << mUrl;
  urlStream << "?SERVICE=" << mServiceType;
  urlStream << "&VERSION=" << mSupportedVersion;
  urlStream << "&REQUEST=GetCapabilities";
  return urlStream;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csl::ogc