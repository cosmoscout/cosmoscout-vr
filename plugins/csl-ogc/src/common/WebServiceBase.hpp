////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSL_OGC_WEB_SERVICE_BASE_HPP
#define CSL_OGC_WEB_SERVICE_BASE_HPP

#include "csl_ogc_export.hpp"

#include "OGCExceptionReport.hpp"

#include <VistaTools/tinyXML/tinyxml.h>
#include <nlohmann/json.hpp>
#include <optional>

namespace csl::ogc {

class CSL_OGC_EXPORT WebServiceBase {
 protected:
  struct TagNames {
    std::string mCapabilitiesRoot;
  };

 public:
  /// Possible modes for cache usage when loading capabilities.
  enum class CacheMode {
    /// Always use available cached files without checking if they are up to date.
    eAlways,
    /// Check if cached files are up to date using their update sequence.
    eUpdateSequence,
    /// Never use cached files, always request new capabilities from the server.
    eNever
  };

  /// Create an object for a web service accessible at the given URL.
  /// The url string should be the base URL of the web service without a query string.
  /// cacheMode can be used to control the caching behavior for the capability document.
  /// If caching is activated, cacheDir should be the path to a directory which can be
  /// used for caching.
  WebServiceBase(std::string url, CacheMode cacheMode, std::string cacheDir,
      std::string serviceType, std::string supportedVersion, TagNames tagNames);

  /// Gets the base URL of the service
  std::string const& getUrl() const noexcept;

  /// Gets a brief description of the service.
  std::string const& getTitle() const noexcept;

  virtual ~WebServiceBase() = default;

 protected:
  /// Requests the capabilities and parses them into a xml document
  VistaXML::TiXmlElement* getCapabilities();

  std::stringstream getGetCapabilitiesUrl() const noexcept;

  virtual std::unique_ptr<OGCExceptionReport> createExceptionReport(
      VistaXML::TiXmlDocument const& doc) const = 0;

  void setTitle(std::string title) noexcept;

 private:
  /// Tries to load cached capabilities for this web service.
  std::optional<VistaXML::TiXmlDocument> getCapabilitiesFromCache();

  /// Requests a new capability document from the server.
  /// Returns the document as a parsed TiXmlDocument and as a raw string for caching.
  std::tuple<VistaXML::TiXmlDocument, std::string> requestCapabilities();

  /// Checks if the given capabilities document is up to date.
  /// The returned optional either contains the current capability document as a parsed
  /// TiXmlDocument and optionally as a raw string, or is empty if a new capability document should
  /// be requested. If the returned raw string is not empty, the returned capability document may be
  /// different to the one given to this function and thus should be saved to the cache.
  std::optional<std::tuple<VistaXML::TiXmlDocument, std::optional<std::string>>>
  checkUpdateSequence(VistaXML::TiXmlDocument cacheDoc);

  std::string       mUrl{};
  const CacheMode   mCacheMode{};
  const std::string mCacheDir{};
  const std::string mCacheFileName{};

  std::string mTitle{};

  const std::string mServiceType{};
  const std::string mSupportedVersion{};

  const TagNames mTagNames;

  std::optional<VistaXML::TiXmlDocument> mDoc;
};

NLOHMANN_JSON_SERIALIZE_ENUM(
    WebServiceBase::CacheMode, {
                                   {WebServiceBase::CacheMode::eAlways, "always"},
                                   {WebServiceBase::CacheMode::eUpdateSequence, "updateSequence"},
                                   {WebServiceBase::CacheMode::eNever, "never"},
                               })

} // namespace csl::ogc

#endif // CSL_OGC_WEB_SERVICE_BASE_HPP