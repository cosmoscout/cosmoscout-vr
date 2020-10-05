////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CSP_VR_ACCESSIBILITY_PLUGIN_HPP
#define CSP_VR_ACCESSIBILITY_PLUGIN_HPP

#include "../../../src/cs-core/PluginBase.hpp"
#include "../../../src/cs-core/Settings.hpp"
#include "../../../src/cs-utils/Property.hpp"

namespace csp::vraccessibility {

class FloorGrid;

/// This plugin adds a floor grid. The grid is rendered below the observer.
/// The configuration of this plugin is done via the provided json config.
/// See README.md for details.
class Plugin : public cs::core::PluginBase {
 public:
  struct Settings {
    /// Toggle, whether the grid is hidden (false) or visible (true)
    cs::utils::DefaultProperty<bool> mEnabled{true};

    /// The size of the grid, i.e. mesh size, scale factor (default = 1)
    cs::utils::DefaultProperty<float> mSize{1.0F};

    /// The height offset to adjust the grid to the floor
    cs::utils::DefaultProperty<float> mOffset{-1.8F};

    /// The falloff distance when the grid fades.
    cs::utils::DefaultProperty<float> mFalloff{100.0F};

    /// The texture used for the grid (b/w texture)
    cs::utils::DefaultProperty<std::string> mTexture{"../share/resources/textures/gridCentered.png"};

    /// The opacity of the grid (default 1, fully opaque, to 0, fully transparent)
    cs::utils::DefaultProperty<float> mAlpha{1.0F};

    /// The color of the grid (default white #FFFFFF)
    cs::utils::DefaultProperty<std::string> mColor{"#FFFFFF"};
  };

  void init() override;
  void deInit() override;

  void update() override;

 private:
  void onLoad();

  std::shared_ptr<Settings> mPluginSettings = std::make_shared<Settings>();
  std::shared_ptr<FloorGrid> mGrid;
  bool resetColorPicker{true};

  int mOnLoadConnection = -1;
  int mOnSaveConnection = -1;
};

} // namespace csp::floorgrid

#endif // CSP_FLOOR_GRID_PLUGIN_HPP
