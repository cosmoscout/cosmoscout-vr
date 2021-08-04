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
    /// Toggle, whether the grid is hidden (false) or visible (true).
    cs::utils::DefaultProperty<bool> mEnabled{true};

    /// The size of the grid, i.e. mesh size, scale factor (default = 1).
    cs::utils::DefaultProperty<float> mSize{1.0F};

    /// The height offset to adjust the grid to the floor.
    cs::utils::DefaultProperty<float> mOffset{-1.8F};

    /// The falloff distance when the grid fades.
    cs::utils::DefaultProperty<float> mFalloff{100.0F};

    /// The texture used for the grid (b/w texture).
    cs::utils::DefaultProperty<std::string> mTexture{
        "../share/resources/textures/gridCentered.png"};

    /// The opacity of the grid (default: 1, fully opaque, to 0, fully transparent).
    cs::utils::DefaultProperty<float> mAlpha{1.0F};

    /// The color of the grid (default: white #FFFFFF).
    cs::utils::DefaultProperty<std::string> mColor{"#FFFFFF"};

    /// Toggle, whether the FoV Vignette is used or not.
    cs::utils::DefaultProperty<bool> mFovVignetteEnabled{true};

    /// Toggle, whether the FoV Vignette is always drawn.
    cs::utils::DefaultProperty<bool> mFovVignetteDebug{false};

    /// The inner radius of the FoV Vignette (distance from center to rim where the gradient
    /// starts)(0, no radius, to 1, the edges of the screen).
    cs::utils::DefaultProperty<float> mFovVignetteInnerRadius{0.5F};

    /// The outer radius of the FoV Vignette (distance from start of the gradient to end of
    /// gradient).
    cs::utils::DefaultProperty<float> mFovVignetteOuterRadius{1.0F};

    /// The color of the FoV Vignette (default: white #FFFFFF).
    cs::utils::DefaultProperty<std::string> mFovVignetteColor{"#FFFFFF"};

    /// The duration of the fade animation (in seconds).
    cs::utils::DefaultProperty<double> mFovVignetteFadeDuration{1.0};

    /// The deadzone of the fade animation where the animation is not played on small actions (in
    /// seconds).
    cs::utils::DefaultProperty<double> mFovVignetteFadeDeadzone{0.5};

    /// The threshold velocity (0 to ~10 = max. speed from movement controls) below which the
    /// vignette is not triggered.
    cs::utils::DefaultProperty<float> mFovVignetteLowerVelocityThreshold{0.2F};

    /// The threshold velocity (0 to ~10 = max. speed from movement controls) above which the
    /// vignette is set to the above defined radii.
    cs::utils::DefaultProperty<float> mFovVignetteUpperVelocityThreshold{10.0F};

    /// The toggle to use dynamic radius adjustment instead of fading the vignette in above
    /// threshold.
    cs::utils::DefaultProperty<bool> mFovVignetteUseDynamicRadius{false};

    /// The toggle to use only vertical vignetting.
    cs::utils::DefaultProperty<bool> mFovVignetteUseVerticalOnly{false};
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
