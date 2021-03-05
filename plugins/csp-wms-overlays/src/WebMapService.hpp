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
  /// Struct for storing general WMS settings.
  struct Settings {
    std::optional<int> mMaxWidth;
    std::optional<int> mMaxHeight;
  };

  /// Create an object for a WMS accessible at the given URL.
  /// The url string should be the base URL of the WMS without a query string.
  /// cacheDir should be the path to a directory which will be used for
  /// caching capability documents.
  WebMapService(std::string url, std::string cacheDir);

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

  /// The returned optional either contains the current capability document as a raw string and as a
  /// parsed TiXmlDocument, or is empty if a new capability document should be requested.
  /// The returned capability document is not necessarily equal to the document found in cache.
  /// The boolean flag specifies whether the file should be cached.
  std::optional<std::tuple<std::string, VistaXML::TiXmlDocument, bool>> getCapabilitiesFromCache();

  std::stringstream getGetCapabilitiesUrl() const;

  std::optional<VistaXML::TiXmlDocument> mDoc;

  const std::string mUrl;
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
