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

/// Class for storing information on a single layer of a WMS.
class WebMapLayer {
 public:
  /// Struct for storing information on a single style for a WMS layer.
  struct Style {
    const std::string                mTitle;     ///< Human readable description of the style.
    const std::string                mName;      ///< Internal name of the style for requests.
    const std::optional<std::string> mLegendUrl; ///< URL at which a legend image may be found.

    Style(VistaXML::TiXmlElement* element);

   private:
    std::optional<std::string> getLegendUrl(VistaXML::TiXmlElement* element);
  };

  /// Struct for storing general layer settings.
  struct Settings {
    bool mNoSubsets =
        false; ///< If true, no different bounds than the default ones may be requested.
    std::vector<std::string>
                               mCrs; ///< List of coordinate reference systems for which data is available.
    std::optional<int>         mFixedWidth;  ///< Only textures with this width may be requested.
    std::optional<int>         mFixedHeight; ///< Only textures with this height may be requested.
    Bounds                     mBounds;      ///< Default (maximum) bounds of the layer.
    bool                       mOpaque = false; ///< Specifies whether the layer is opaque.
    std::vector<Style>         mStyles;         ///< List of styles for the layer.
    std::vector<TimeInterval>  mTimeIntervals;  ///< TimeIntervals, for which data is available.
    std::optional<std::string> mAttribution;    ///< Attribution for the layer.
    std::optional<double>
        mMinScale; ///< Minimum scale denominator for which it is appropriate to generate a map.
    std::optional<double>
        mMaxScale; ///< Maximum scale denominator for which it is appropriate to generate a map.
  };

  WebMapLayer(VistaXML::TiXmlElement* element, Settings settings);

  /// Gets a human readable description of the layer.
  std::string getTitle() const;
  /// Gets the internal name of the layer used for requests.
  std::string getName() const;
  /// Gets a narrative description of the layer.
  std::optional<std::string> getAbstract() const;
  /// Gets the general settings of the layer.
  Settings getSettings() const;

  /// Checks if map data may be requested for the layer.
  bool isRequestable() const;
  /// Gets a list of all child layers of the layer.
  std::vector<WebMapLayer> getAllLayers() const;
  /// Adds all child layers to the layers list, for which map data may be requested.
  void getRequestableLayers(std::vector<WebMapLayer>& layers) const;

 private:
  std::string                mTitle;
  std::optional<std::string> mName;
  std::optional<std::string> mAbstract;

  std::vector<WebMapLayer> mSubLayers;

  Settings mSettings;
};

} // namespace csp::wmsoverlays

#endif // CSP_WMS_OVERLAYS_WEB_MAP_LAYER_HPP
