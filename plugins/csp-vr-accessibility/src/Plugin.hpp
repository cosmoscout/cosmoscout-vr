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
class FovVignette;

/// This plugin adds VR accessibility options like a floor grid and a FoV vignette.
/// The configuration of this plugin is done via the provided json config.
/// See README.md for details.
class Plugin : public cs::core::PluginBase {
 public:
  struct Settings {
    struct Grid {
      /// Toggle, whether the grid is hidden (false) or visible (true).
      cs::utils::DefaultProperty<bool> mEnabled{true};

      /// The scale of the grid texture.
      cs::utils::DefaultProperty<float> mSize{0.5F};

      /// The height offset to adjust the grid to the floor.
      cs::utils::DefaultProperty<float> mOffset{-1.8F};

      /// The size of the grid.
      cs::utils::DefaultProperty<float> mExtent{10.0F};

      /// The texture used for the grid (b/w texture).
      cs::utils::DefaultProperty<std::string> mTexture{
          "../share/resources/textures/gridCentered.png"};

      /// The opacity of the grid (default: 1, fully opaque, to 0, fully transparent).
      cs::utils::DefaultProperty<float> mAlpha{1.0F};

      /// The color of the grid.
      cs::utils::DefaultProperty<std::string> mColor{"#FFFFFF"};
    };

    struct Vignette {
      /// Toggle, whether the FoV Vignette is used or not.
      cs::utils::DefaultProperty<bool> mEnabled{true};

      /// Toggle, whether the FoV Vignette is always drawn.
      cs::utils::DefaultProperty<bool> mDebug{false};

      /// The inner and the outer radius of the FoV Vignette.
      cs::utils::DefaultProperty<glm::vec2> mRadii{glm::vec2(0.5F, 1.0F)};

      /// The color of the FoV Vignette.
      cs::utils::DefaultProperty<std::string> mColor{"#000000"};

      /// The duration of the fade animation (in seconds).
      cs::utils::DefaultProperty<double> mFadeDuration{1.0};

      /// The deadzone of the fade animation where the animation is not played on small actions (in
      /// seconds).
      cs::utils::DefaultProperty<double> mFadeDeadzone{0.5};

      /// If the observer speed is within this range, the vignette will fade in or out.
      cs::utils::DefaultProperty<glm::vec2> mVelocityThresholds{glm::vec2(0.2F, 10.0F)};

      /// The toggle to use dynamic radius adjustment instead of fading the vignette in above
      /// threshold.
      cs::utils::DefaultProperty<bool> mUseDynamicRadius{false};

      /// The toggle to use only vertical vignetting.
      cs::utils::DefaultProperty<bool> mUseVerticalOnly{false};
    };

    /// The container for the Grid Settings
    Plugin::Settings::Grid mGridSettings;

    /// The container for the Vignette Settings
    Plugin::Settings::Vignette mVignetteSettings;
  };

  void init() override;
  void deInit() override;

  void update() override;

  static glm::vec4 GetColorFromHexString(std::string color);

 private:
  void onLoad();

  std::shared_ptr<Settings>    mPluginSettings = std::make_shared<Settings>();
  std::shared_ptr<FloorGrid>   mGrid;
  std::shared_ptr<FovVignette> mVignette;

  bool resetColorPicker{true};

  int mOnLoadConnection = -1;
  int mOnSaveConnection = -1;
};

} // namespace csp::vraccessibility

#endif // CSP_FLOOR_GRID_PLUGIN_HPP
