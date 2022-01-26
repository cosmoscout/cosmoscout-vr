////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2020 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CSP_WCS_OVERLAYS_WEB_COVERAGE_SERVICE_HPP
#define CSP_WCS_OVERLAYS_WEB_COVERAGE_SERVICE_HPP

#include "WebCoverage.hpp"

#include <VistaTools/tinyXML/tinyxml.h>

#include <array>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace csp::wcsoverlays {
class WebCoverageService {
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

  /// Create an object for a WCS accessible at the given URL.
  /// The url string should be the base URL of the WCS without a query string.
  /// cacheMode can be used to control the caching behavior for the capability document.
  /// If caching is activated, cacheDir should be the path to a directory which can be
  /// used for caching.
  WebCoverageService(std::string url, CacheMode cacheMode, std::string cacheDir);

  /// Gets the base URL of the service
  std::string const& getUrl() const;
  /// Gets a brief description of the service.
  std::string const& getTitle() const;

  /// Gets a list of all coverages of the service, for which a coverage can be requested.
  std::vector<WebCoverage> const& getCoverages() const;
  /// Gets the coverage with the given title or coverage id, if one exists.
  /// Returns an empty optional otherwise.
  std::optional<WebCoverage> getCoverage(std::string const& titleOrId) const;

 protected:
  /// Requests the capabilities and parses them into a xml document
  VistaXML::TiXmlElement* getCapabilities();

  /// Parses the servers title
  std::string parseTitle();

  /// Tries to load cached capabilities for this WCS.
  std::optional<VistaXML::TiXmlDocument> getCapabilitiesFromCache();
  /// Tries to load cached capabilities for this WCS.
  /// std::optional<VistaXML::TiXmlDocument> getCapabilitiesFromCache();
  /// Checks if the given capabilities document is up to date.
  /// The returned optional either contains the current capability document as a parsed
  /// TiXmlDocument and optionally as a raw string, or is empty if a new capability document should
  /// be requested. If the returned raw string is not empty, the returned capability document may be
  /// different to the one given to this function and thus should be saved to the cache.
  std::optional<std::tuple<VistaXML::TiXmlDocument, std::optional<std::string>>>
  checkUpdateSequence(VistaXML::TiXmlDocument cacheDoc);
  /// Requests a new capability document from the server.
  /// Returns the document as a parsed TiXmlDocument and as a raw string for caching.
  std::tuple<VistaXML::TiXmlDocument, std::string> requestCapabilities();

  /// Builds the url to where the capabilities are found
  std::stringstream getGetCapabilitiesUrl() const;

  /// The XML representation of the WCS servers capabilities
  std::optional<VistaXML::TiXmlDocument> mDoc;

  /// The url to the wcs service, is used for generating links to GetCapabilities etc.
  const std::string mUrl;
  const CacheMode   mCacheMode;
  const std::string mCacheDir;
  const std::string mCacheFileName;

  /// The WCS servers title
  const std::string mTitle;

  std::vector<WebCoverage> mRequestableCoverages;

 private:
  void parseCoverages();
};
} // namespace csp::wcsoverlays
#endif // CSP_WCS_OVERLAYS_WEB_COVERAGE_SERVICE_HPP
