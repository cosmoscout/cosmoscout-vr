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

class WebMapService {
 public:
  struct Settings {
    std::optional<int> mMaxWidth;
    std::optional<int> mMaxHeight;
  };

  WebMapService(std::string url, std::string cacheDir);

  std::string getUrl() const;
  std::string getTitle() const;

  Settings getSettings() const;

  WebMapLayer                getRootLayer() const;
  std::vector<WebMapLayer>   getLayers() const;
  std::optional<WebMapLayer> getLayer(std::string name) const;

  bool isFormatSupported(std::string format) const;

 private:
  VistaXML::TiXmlElement*  getCapabilities();
  WebMapLayer              parseRootLayer();
  std::string              parseTitle();
  Settings                 parseSettings();
  std::vector<std::string> parseMapFormats();

  std::optional<std::pair<std::string, VistaXML::TiXmlDocument>> getCapabilitiesFromCache();

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
