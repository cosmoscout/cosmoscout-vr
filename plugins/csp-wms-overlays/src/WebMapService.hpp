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
#include <optional>
#include <string>
#include <vector>
#include <memory>

namespace csp::wmsoverlays {

class WebMapService {
 public:
  WebMapService(std::string url);

	std::vector<WebMapLayer> getLayers();

 private:
  VistaXML::TiXmlDocument getCapabilities();

  std::string mUrl;

  std::unique_ptr<WebMapLayer> mRootLayer;
};

} // namespace csp::wmsoverlays

#endif // CSP_WMS_OVERLAYS_WEB_MAP_SERVICE_HPP
