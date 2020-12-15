////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CSP_WMS_OVERLAYS_PLUGIN_HPP
#define CSP_WMS_OVERLAYS_PLUGIN_HPP

#include "utils.hpp"

#include "../../../src/cs-core/PluginBase.hpp"
#include "../../../src/cs-utils/DefaultProperty.hpp"

#include <map>
#include <string>

namespace csp::wmsoverlays {

class TextureOverlayRenderer;

/// This plugin provides the rendering of planets as spheres with a texture and an additional WMS
/// based texture. Despite its name it can also render moons :P. It can be configured via the
/// applications config file. See README.md for details.
class Plugin : public cs::core::PluginBase {
 public:
  /// The startup settings of the plugin.
  struct Settings {
    /// Specifies whether to interpolate textures between timesteps (does not work when pre-fetch is
    /// inactive).
    cs::utils::DefaultProperty<bool> mEnableInterpolation{true};

    /// Specifies whether to display timespan.
    /// Needs to be specified true in config for a data set which enables it in order to be used.
    cs::utils::DefaultProperty<bool> mEnableTimespan{false};

    /// Path to the map cache folder, can be absolute or relative to the cosmoscout executable.
    cs::utils::DefaultProperty<std::string> mMapCache{"texture-cache"};

    /// A single WMS data set.
    struct WMSConfig {
      std::string mCopyright; ///< The copyright holder of the data set (also shown in the UI).
      std::string mUrl;       ///< The URL of the map server including the "SERVICE=wms" parameter.
      std::string mFormat;    ///< Download image file format: png or jpeg.
      int         mWidth;     ///< The width of the WMS image.
      int         mHeight;    ///< The height of the WMS image.
      std::optional<std::string> mTime;   ///< Time intervals of WMS images.
      std::string                mLayers; ///< A comma,seperated list of WMS layers.
      std::optional<int>
          mPrefetchCount; ///< The amount of textures that gets pre-fetched in every time direction.
      std::optional<bool> mTimespan; ///< True if the WMS server enables the use of timespan.

      cs::utils::DefaultProperty<std::array<double, 2>> mLatRange{{-90., 90.}};
      cs::utils::DefaultProperty<std::array<double, 2>> mLonRange{{-180., 180.}};
    };

    /// The startup settings for a planet.
    struct SimpleWMSBody {
      std::optional<int> mGridResolutionX;   ///< The x resolution of the body grid.
      std::optional<int> mGridResolutionY;   ///< The y resolution of the body gird.
      std::string        mTexture;           ///< The path to surface texture.
      std::string        mActiveWMS;         ///< The name of the currently active WMS data set.
      std::map<std::string, WMSConfig> mWMS; ///< The data sets containing WMS data.
    };

    std::map<std::string, SimpleWMSBody> mBodies; ///< A list of bodies with their anchor names.
  };

  void init() override;
  void deInit() override;

 private:
  void onLoad();

  Settings::SimpleWMSBody& getBodySettings(
      std::shared_ptr<TextureOverlayRenderer> const& wmsOverlay) const;
  void setWMSSource(
      std::shared_ptr<TextureOverlayRenderer> const& wmsOverlay, std::string const& name) const;

  std::shared_ptr<Settings> mPluginSettings = std::make_shared<Settings>();
  std::map<std::string, std::shared_ptr<TextureOverlayRenderer>> mWMSOverlays;

  int mActiveBodyConnection = -1;
  int mOnLoadConnection     = -1;
  int mOnSaveConnection     = -1;
};

} // namespace csp::wmsoverlays

#endif // CSP_WMS_OVERLAYS_PLUGIN_HPP
