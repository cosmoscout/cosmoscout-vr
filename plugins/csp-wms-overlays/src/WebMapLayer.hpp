////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2020 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CSP_WMS_OVERLAYS_WEB_MAP_LAYER_HPP
#define CSP_WMS_OVERLAYS_WEB_MAP_LAYER_HPP

#include "utils.hpp"

#include <VistaTools/tinyXML/tinyxml.h>

#include <array>
#include <optional>
#include <string>
#include <vector>

namespace csp::wmsoverlays {

class WebMapLayer {
 public:
  struct Style {
    const std::string                mTitle;
    const std::string                mName;
    const std::optional<std::string> mLegendUrl;

    Style(VistaXML::TiXmlElement* element);

   private:
    std::optional<std::string> getLegendUrl(VistaXML::TiXmlElement* element);
  };

  struct Settings {
    bool                       mOpaque    = false;
    bool                       mNoSubsets = false;
    std::optional<int>         mFixedWidth;
    std::optional<int>         mFixedHeight;
    Bounds                     mBounds;
    std::vector<TimeInterval>  mTimeIntervals;
    std::vector<Style>         mStyles;
    std::vector<std::string>   mCrs;
    // TODO Other dimensions?
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
