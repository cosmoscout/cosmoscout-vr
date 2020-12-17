////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2020 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CSP_WMS_OVERLAYS_WEB_MAP_LAYER_HPP
#define CSP_WMS_OVERLAYS_WEB_MAP_LAYER_HPP

#include <VistaTools/tinyXML/tinyxml.h>

#include <array>
#include <optional>
#include <string>
#include <vector>

namespace csp::wmsoverlays {

class WebMapLayer {
 public:
  struct Settings {
    bool                       mOpaque    = false;
    bool                       mNoSubsets = false;
    std::optional<int>         mFixedWidth;
    std::optional<int>         mFixedHeight;
    std::array<double, 2>      mLonRange = {-180., 180.};
    std::array<double, 2>      mLatRange = {-90., 90.};
    std::optional<std::string> mTime;
    // TODO Crs
    // TODO Other dimensions?
    // TODO Styles + Legends
    std::optional<std::string> mAttribution;
  };

  WebMapLayer(VistaXML::TiXmlElement* element, Settings settings);

  std::string getTitle() const;
  std::string getName() const;
  Settings    getSettings() const;

  bool isRequestable();
  void getRequestableLayers(std::vector<WebMapLayer>& layers);

 private:
  std::string                mTitle;
  std::optional<std::string> mName;

  std::vector<WebMapLayer> mSubLayers;

  Settings mSettings;
};

} // namespace csp::wmsoverlays

#endif // CSP_WMS_OVERLAYS_WEB_MAP_LAYER_HPP
