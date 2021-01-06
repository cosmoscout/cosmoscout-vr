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
  WebMapService(std::string url, std::string cacheDir);

  std::string getUrl() const;
  std::string getTitle() const;

  std::vector<WebMapLayer>   getLayers() const;
  std::optional<WebMapLayer> getLayer(std::string name) const;

 private:
  VistaXML::TiXmlElement* getCapabilities();
  WebMapLayer             parseRootLayer();
  std::string             parseTitle();

  std::optional<std::pair<std::string, VistaXML::TiXmlDocument>> getCapabilitiesFromCache();

  std::stringstream getGetCapabilitiesUrl();

  std::optional<VistaXML::TiXmlDocument> mDoc;

  const std::string mUrl;
  const std::string mCacheDir;
  const std::string mCacheFileName;

  const std::string mTitle;

  WebMapLayer              mRootLayer;
  std::vector<WebMapLayer> mRequestableLayers;
};

} // namespace csp::wmsoverlays

#endif // CSP_WMS_OVERLAYS_WEB_MAP_SERVICE_HPP
