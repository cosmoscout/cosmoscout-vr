////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2020 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CSP_WMS_OVERLAYS_WEB_MAP_SERVICE_HPP
#define CSP_WMS_OVERLAYS_WEB_MAP_SERVICE_HPP

#include "WebMapLayer.hpp"

#include <VistaTools/tinyXML/tinyxml.h>

#include <array>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace csp::wmsoverlays {

/// Class for storing information on a Web Map Service.
class WebMapService {
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

  /// Struct for storing general WMS settings.
  struct Settings {
    std::optional<int> mMaxWidth;
    std::optional<int> mMaxHeight;
  };

  /// Create an object for a WMS accessible at the given URL.
  /// The url string should be the base URL of the WMS without a query string.
  /// cacheMode can be used to control the caching behavior for the capability document.
  /// If caching is activated, cacheDir should be the path to a directory which can be
  /// used for caching.
  WebMapService(std::string url, CacheMode cacheMode, std::string cacheDir);

  /// Gets the base URL of the service
  std::string const& getUrl() const;
  /// Gets a brief description of the service.
  std::string const& getTitle() const;

  /// Gets the general settings of the service.
  Settings const& getSettings() const;

  /// Gets the root layer of the service.
  WebMapLayer const& getRootLayer() const;
  /// Gets a list of all layers of the service, for which maps can be requested.
  std::vector<WebMapLayer> const& getLayers() const;
  /// Gets the layer with the given name, if one exists.
  /// Returns an empty optional otherwise.
  std::optional<WebMapLayer> getLayer(std::string const& name) const;

  /// Checks if the service can return maps of the given MIME type.
  bool isFormatSupported(std::string const& format) const;

 private:
  VistaXML::TiXmlElement*  getCapabilities();
  WebMapLayer              parseRootLayer();
  std::string              parseTitle();
  Settings                 parseSettings();
  std::vector<std::string> parseMapFormats();

  /// Tries to load cached capabilities for this WMS.
  std::optional<VistaXML::TiXmlDocument> getCapabilitiesFromCache();
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

  std::stringstream getGetCapabilitiesUrl() const;

  std::optional<VistaXML::TiXmlDocument> mDoc;

  const std::string mUrl;
  const CacheMode   mCacheMode;
  const std::string mCacheDir;
  const std::string mCacheFileName;

  const std::string mTitle;
  const Settings    mSettings;

  const std::vector<std::string> mMapFormats;

  WebMapLayer              mRootLayer;
  std::vector<WebMapLayer> mRequestableLayers;
};

} // namespace csp::wmsoverlays

#endif // CSP_WMS_OVERLAYS_WEB_MAP_SERVICE_HPP
