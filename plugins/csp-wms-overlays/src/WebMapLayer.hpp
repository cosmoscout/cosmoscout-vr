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
    /// Internal name of the style for requests.
    const std::string mName;
    /// Human readable description of the style.
    const std::string mTitle;
    /// URL at which a legend image may be found.
    const std::optional<std::string> mLegendUrl;

    explicit Style(VistaXML::TiXmlElement* element);

   private:
    static std::optional<std::string> getLegendUrl(VistaXML::TiXmlElement* element);
  };

  /// Struct for storing general layer settings.
  struct Settings {
    /// If true, no different bounds than the default ones may be requested.
    bool mNoSubsets = false;
    /// List of coordinate reference systems for which data is available.
    std::vector<std::string> mCrs;
    /// Only textures with this width may be requested.
    std::optional<int> mFixedWidth;
    /// Only textures with this height may be requested.
    std::optional<int> mFixedHeight;
    /// Default (maximum) bounds of the layer.
    Bounds mBounds;
    /// Specifies whether the layer is opaque.
    bool mOpaque = false;
    /// List of styles for the layer.
    std::vector<Style> mStyles;
    /// TimeIntervals, for which data is available.
    std::vector<TimeInterval> mTimeIntervals;
    /// Attribution for the layer.
    std::optional<std::string> mAttribution;
    /// Minimum scale denominator for which it is appropriate to generate a map.
    std::optional<double> mMinScale;
    /// Maximum scale denominator for which it is appropriate to generate a map.
    std::optional<double> mMaxScale;
  };

  WebMapLayer(VistaXML::TiXmlElement* element, Settings settings);

  /// Gets a human readable description of the layer.
  std::string const& getTitle() const;
  /// Gets the internal name of the layer used for requests.
  std::string getName() const;
  /// Gets a narrative description of the layer.
  std::optional<std::string> const& getAbstract() const;
  /// Gets the general settings of the layer.
  Settings const& getSettings() const;

  /// Checks if map data may be requested for the layer.
  bool isRequestable() const;
  /// Gets a list of all child layers of the layer.
  std::vector<WebMapLayer> const& getAllLayers() const;
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
