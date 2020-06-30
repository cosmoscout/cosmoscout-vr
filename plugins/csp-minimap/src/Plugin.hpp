////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CSP_MINIMAP_PLUGIN_HPP
#define CSP_MINIMAP_PLUGIN_HPP

#include "../../../src/cs-core/PluginBase.hpp"
#include "../../../src/cs-core/Settings.hpp"
#include "../../../src/cs-scene/CelestialBody.hpp"

#include <map>
#include <vector>

namespace csp::minimap {

/// This adds a configurable 2D-Minimap to the user interface. It shows the current observer
/// position, the bookmarks for the currently active body and allows adding new bookmarks.
class Plugin : public cs::core::PluginBase {
 public:
  struct Settings {
    enum class ProjectionType { eNone, eMercator, eEquirectangular };
    enum class LayerType { eNone, eWMS, eWMTS };

    struct Map {
      /// Either mercator or equirectangular.
      ProjectionType mProjection;

      /// Either WMS or WMTS. If WMS, you have to give a "layer" parameter in the config object.
      LayerType mType;

      /// WMTS template URL or WMS service endpoint, including the '?'.
      std::string mURL;

      /// This is directly passed to Leaflet, the supported options are documented on the page
      /// linked below. Note that if mType is eWMS all additional options will be sent to the WMS
      /// server as extra parameters in each request URL.
      /// https://leafletjs.com/reference-1.6.0.html#tilelayer
      /// https://leafletjs.com/reference-1.6.0.html#tilelayer-wms
      nlohmann::json mConfig;
    };

    /// This maps a minimap configuration to a SPICE center name. The minimap will be configured
    /// according to the currently active body.
    std::map<std::string, Map> mMaps;

    /// If no configuration is found in the map above, this configuration will be used instead.
    std::optional<Map> mDefaultMap;
  };

  void init() override;
  void deInit() override;

 private:
  void onAddBookmark(std::shared_ptr<cs::scene::CelestialBody> const& activeBody,
      uint32_t bookmarkID, cs::core::Settings::Bookmark const& bookmark);
  void onLoad();

  Settings mPluginSettings;

  int mActiveBodyConnection        = -1;
  int mOnBookmarkAddedConnection   = -1;
  int mOnBookmarkRemovedConnection = -1;
  int mOnLoadConnection            = -1;
  int mOnSaveConnection            = -1;
};

} // namespace csp::minimap

#endif // CSP_MINIMAP_PLUGIN_HPP
