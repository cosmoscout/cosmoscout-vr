////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CSP_SIMPLE_WMS_BODIES_PLUGIN_HPP
#define CSP_SIMPLE_WMS_BODIES_PLUGIN_HPP

#include "utils.hpp"

#include "../../../src/cs-core/PluginBase.hpp"
#include "../../../src/cs-utils/DefaultProperty.hpp"

#include <map>
#include <string>

namespace csp::simplewmsbodies {

class SimpleWMSBody;

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
    cs::utils::DefaultProperty<bool> mEnableTimespan{false};

    /// Path to the map cache folder, can be absolute or relative to the cosmoscout executable.
    cs::utils::DefaultProperty<std::string> mMapCache{"texture-cache"};

    /// A single WMS data set.
    struct WMSConfig {
      std::string mCopyright; ///< The copyright holder of the data set (also shown in the UI).
      std::string mUrl;       ///< The URL of the map server including the "SERVICE=wms" parameter.
      int         mWidth;     ///< The width of the WMS image.
      int         mHeight;    ///< The height of the WMS image.
      std::optional<std::string> mTime;   ///< Time intervals of WMS images.
      std::string                mLayers; ///< A comma,seperated list of WMS layers.
      std::optional<int>
                          mPrefetchCount; ///< The amount of textures that gets pre-fetched in every time direction.
      std::optional<bool> mTimespan; ///< True if the WMS server enables the use of timespan.
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

  Settings::SimpleWMSBody& getBodySettings(std::shared_ptr<SimpleWMSBody> const& body) const;
  void setWMSSource(std::shared_ptr<SimpleWMSBody> const& body, std::string const& name) const;

  /// Add bookmarks to the timeline from time intervals of the current data set.
  void addBookmarks(std::vector<TimeInterval> timeIntervals, std::string wmsName,
      std::string planetName, std::string frameName);

  /// Remove the current bookmarks.
  void removeBookmarks();

  std::shared_ptr<Settings> mPluginSettings = std::make_shared<Settings>();
  std::map<std::string, std::shared_ptr<SimpleWMSBody>> mSimpleWMSBodies;
  std::vector<int>                                      mBookmarkIDs;

  int mActiveBodyConnection = -1;
  int mOnLoadConnection     = -1;
  int mOnSaveConnection     = -1;
};

} // namespace csp::simplewmsbodies

#endif // CSP_SIMPLE_WMS_BODIES_PLUGIN_HPP
