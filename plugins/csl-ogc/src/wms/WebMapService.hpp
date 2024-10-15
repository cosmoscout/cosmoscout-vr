////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSL_OGC_WEB_MAP_SERVICE_HPP
#define CSL_OGC_WEB_MAP_SERVICE_HPP

#include "csl_ogc_export.hpp"

#include "../common/WebServiceBase.hpp"
#include "WebMapLayer.hpp"

#include <VistaTools/tinyXML/tinyxml.h>

#include <array>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace csl::ogc {

/// Class for storing information on a Web Map Service.
class CSL_OGC_EXPORT WebMapService : public WebServiceBase {
 public:
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

 protected:
  std::unique_ptr<OGCExceptionReport> createExceptionReport(
      VistaXML::TiXmlDocument const& doc) const override;

 private:
  WebMapLayer              parseRootLayer();
  Settings                 parseSettings();
  std::vector<std::string> parseMapFormats();
  std::string              parseTitle();

  /// Requests a new capability document from the server.
  /// Returns the document as a parsed TiXmlDocument and as a raw string for caching.
  std::tuple<VistaXML::TiXmlDocument, std::string> requestCapabilities();

  std::optional<VistaXML::TiXmlDocument> mDoc;

  const Settings mSettings;

  const std::vector<std::string> mMapFormats;

  WebMapLayer              mRootLayer;
  std::vector<WebMapLayer> mRequestableLayers;
};

} // namespace csl::ogc

#endif // CSL_OGC_WEB_MAP_SERVICE_HPP
