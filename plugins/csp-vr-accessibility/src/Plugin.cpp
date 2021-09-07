////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "Plugin.hpp"

#include <utility>

#include "../../../src/cs-core/GraphicsEngine.hpp"
#include "../../../src/cs-core/GuiManager.hpp"
#include "../../../src/cs-core/Settings.hpp"
#include "../../../src/cs-core/SolarSystem.hpp"
#include "../../../src/cs-utils/logger.hpp"
#include "../../../src/cs-utils/utils.hpp"
#include "FloorGrid.hpp"
#include "FovVignette.hpp"
#include "logger.hpp"

////////////////////////////////////////////////////////////////////////////////////////////////////

EXPORT_FN cs::core::PluginBase* create() {
  return new csp::vraccessibility::Plugin;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

EXPORT_FN void destroy(cs::core::PluginBase* pluginBase) {
  delete pluginBase; // NOLINT(cppcoreguidelines-owning-memory)
}

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace csp::vraccessibility {

////////////////////////////////////////////////////////////////////////////////////////////////////

void from_json(nlohmann::json const& j, Plugin::Settings::Grid& o) {
  cs::core::Settings::deserialize(j, "enabled", o.mEnabled);
  cs::core::Settings::deserialize(j, "size", o.mSize);
  cs::core::Settings::deserialize(j, "offset", o.mOffset);
  cs::core::Settings::deserialize(j, "extent", o.mExtent);
  cs::core::Settings::deserialize(j, "texture", o.mTexture);
  cs::core::Settings::deserialize(j, "alpha", o.mAlpha);
  cs::core::Settings::deserialize(j, "color", o.mColor);
}

void to_json(nlohmann::json& j, Plugin::Settings::Grid const& o) {
  cs::core::Settings::serialize(j, "enabled", o.mEnabled);
  cs::core::Settings::serialize(j, "size", o.mSize);
  cs::core::Settings::serialize(j, "offset", o.mOffset);
  cs::core::Settings::serialize(j, "extent", o.mExtent);
  cs::core::Settings::serialize(j, "texture", o.mTexture);
  cs::core::Settings::serialize(j, "alpha", o.mAlpha);
  cs::core::Settings::serialize(j, "color", o.mColor);
}

void from_json(nlohmann::json const& j, Plugin::Settings::Vignette& o) {
  cs::core::Settings::deserialize(j, "enabled", o.mEnabled);
  cs::core::Settings::deserialize(j, "debug", o.mDebug);
  cs::core::Settings::deserialize(j, "innerRadius", o.mInnerRadius);
  cs::core::Settings::deserialize(j, "outerRadius", o.mOuterRadius);
  cs::core::Settings::deserialize(j, "color", o.mColor);
  cs::core::Settings::deserialize(j, "fadeDuration", o.mFadeDuration);
  cs::core::Settings::deserialize(j, "fadeDeadzone", o.mFadeDeadzone);
  cs::core::Settings::deserialize(j, "lowerVelocityThreshold", o.mLowerVelocityThreshold);
  cs::core::Settings::deserialize(j, "upperVelocityThreshold", o.mUpperVelocityThreshold);
  cs::core::Settings::deserialize(j, "useDynamicRadius", o.mUseDynamicRadius);
  cs::core::Settings::deserialize(j, "useVerticalOnly", o.mUseVerticalOnly);
}

void to_json(nlohmann::json& j, Plugin::Settings::Vignette const& o) {
  cs::core::Settings::serialize(j, "enabled", o.mEnabled);
  cs::core::Settings::serialize(j, "debug", o.mDebug);
  cs::core::Settings::serialize(j, "innerRadius", o.mInnerRadius);
  cs::core::Settings::serialize(j, "outerRadius", o.mOuterRadius);
  cs::core::Settings::serialize(j, "color", o.mColor);
  cs::core::Settings::serialize(j, "fadeDuration", o.mFadeDuration);
  cs::core::Settings::serialize(j, "fadeDeadzone", o.mFadeDeadzone);
  cs::core::Settings::serialize(j, "lowerVelocityThreshold", o.mLowerVelocityThreshold);
  cs::core::Settings::serialize(j, "upperVelocityThreshold", o.mUpperVelocityThreshold);
  cs::core::Settings::serialize(j, "useDynamicRadius", o.mUseDynamicRadius);
  cs::core::Settings::serialize(j, "useVerticalOnly", o.mUseVerticalOnly);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void from_json(nlohmann::json const& j, Plugin::Settings& o) {
  cs::core::Settings::deserialize(j, "grid", o.mGridSettings);
  cs::core::Settings::deserialize(j, "vignette", o.mVignetteSettings);
}

void to_json(nlohmann::json& j, Plugin::Settings const& o) {
  cs::core::Settings::serialize(j, "grid", o.mGridSettings);
  cs::core::Settings::serialize(j, "vignette", o.mVignetteSettings);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::init() {
  logger().info("Loading plugin...");

  mOnLoadConnection = mAllSettings->onLoad().connect([this]() { onLoad(); });
  mOnSaveConnection = mAllSettings->onSave().connect(
      [this]() { mAllSettings->mPlugins["csp-vr-accessibility"] = *mPluginSettings; });

  // add settings to GUI
  mGuiManager->addSettingsSectionToSideBarFromHTML(
      "VR Accessibility", "blur_circular", "../share/resources/gui/vr_accessibility_settings.html");
  mGuiManager->addScriptToGuiFromJS("../share/resources/gui/js/csp-vr-accessibility.js");

  // register callback for grid enable grid checkbox
  mGuiManager->getGui()->registerCallback("floorGrid.setEnabled",
      "Enables or disables rendering the grid.",
      std::function([this](bool enable) { mPluginSettings->mGridSettings.mEnabled = enable; }));
  mPluginSettings->mGridSettings.mEnabled.connectAndTouch(
      [this](bool enable) { mGuiManager->setCheckboxValue("floorGrid.setEnabled", enable); });

  // register callback for grid size slider
  mGuiManager->getGui()->registerCallback("floorGrid.setSize",
      "Value scales the grid texture size between.",
      std::function([this](double value) {
        mPluginSettings->mGridSettings.mSize = static_cast<float>(value);
      }));
  mPluginSettings->mGridSettings.mSize.connectAndTouch([this](float value) {
    mGuiManager->setSliderValue("floorGrid.setSize", value);
  });

  // register callback for grid extent slider
  mGuiManager->getGui()->registerCallback("floorGrid.setExtent",
      "Value to scale the entire grid.", std::function([this](double value) {
        mPluginSettings->mGridSettings.mExtent = static_cast<float>(value);
      }));
  mPluginSettings->mGridSettings.mExtent.connectAndTouch(
      [this](float value) { mGuiManager->setSliderValue("floorGrid.setExtent", value); });

  // register callback for grid alpha slider
  mGuiManager->getGui()->registerCallback(
      "floorGrid.setAlpha", "Value to adjust grid opacity.", std::function([this](double value) {
        mPluginSettings->mGridSettings.mAlpha = static_cast<float>(value);
      }));
  mPluginSettings->mGridSettings.mAlpha.connectAndTouch(
      [this](float value) { mGuiManager->setSliderValue("floorGrid.setAlpha", value); });

  // register callback for grid color picker
  mGuiManager->getGui()->registerCallback("floorGrid.setColor",
      "Value to adjust color of the grid.", std::function([this](std::string value) {
        mPluginSettings->mGridSettings.mColor = static_cast<std::string>(value);
      }));
  mPluginSettings->mGridSettings.mColor.connectAndTouch([this](std::string value) {
    mGuiManager->getGui()->callJavascript("CosmoScout.floorGrid.setColorValue", value);
  });

  // register callback for fov vignette enable checkbox
  mGuiManager->getGui()->registerCallback("fovVignette.setEnabled",
      "Enables or disables a Vignette limiting the FoV on movement.",
      std::function([this](bool enable) { mPluginSettings->mVignetteSettings.mEnabled = enable; }));
  mPluginSettings->mVignetteSettings.mEnabled.connectAndTouch(
      [this](bool enable) { mGuiManager->setCheckboxValue("fovVignette.setEnabled", enable); });

  // register callback for fov vignette debug checkbox
  mGuiManager->getGui()->registerCallback("fovVignette.setDebug",
      "Enables or disables the vignette to be drawn permanently.",
      std::function([this](bool enable) { mPluginSettings->mVignetteSettings.mDebug = enable; }));
  mPluginSettings->mVignetteSettings.mDebug.connectAndTouch(
      [this](bool enable) { mGuiManager->setCheckboxValue("fovVignette.setDebug", enable); });

  // register callback for fov vignette use dynamic radius
  mGuiManager->getGui()->registerCallback("fovVignette.setEnableDynamicRadius",
      "Dynamically adjust the radius based on velocity instead of fading the vignette in.",
      std::function(
          [this](bool enable) { mPluginSettings->mVignetteSettings.mUseDynamicRadius = enable; }));
  mPluginSettings->mVignetteSettings.mUseDynamicRadius.connectAndTouch([this](bool enable) {
    mGuiManager->setCheckboxValue("fovVignette.setEnableDynamicRadius", enable);
  });

  // register callback for fov vignette use vertical vignetting only
  mGuiManager->getGui()->registerCallback("fovVignette.setEnableVerticalOnly",
      "Only use a vertical vignette, instead of a circular.", std::function([this](bool enable) {
        mPluginSettings->mVignetteSettings.mUseVerticalOnly = enable;
      }));
  mPluginSettings->mVignetteSettings.mUseVerticalOnly.connectAndTouch([this](bool enable) {
    mGuiManager->setCheckboxValue("fovVignette.setEnableVerticalOnly", enable);
  });

  // register callback for fov vignette inner radius slider
  mGuiManager->getGui()->registerCallback("fovVignette.setInnerRadius",
      "Value to adjust the inner radius (start of gradient) of the vignette.",
      std::function([this](double value) {
        mPluginSettings->mVignetteSettings.mInnerRadius = static_cast<float>(value);
      }));
  mPluginSettings->mVignetteSettings.mInnerRadius.connectAndTouch(
      [this](float value) { mGuiManager->setSliderValue("fovVignette.setInnerRadius", value); });

  // register callback for fov vignette outer radius slider
  mGuiManager->getGui()->registerCallback("fovVignette.setOuterRadius",
      "Value to adjust the outer radius (end of gradient) of the vignette.",
      std::function([this](double value) {
        mPluginSettings->mVignetteSettings.mOuterRadius = static_cast<float>(value);
      }));
  mPluginSettings->mVignetteSettings.mOuterRadius.connectAndTouch(
      [this](float value) { mGuiManager->setSliderValue("fovVignette.setOuterRadius", value); });

  // register callback for fov vignette color picker
  mGuiManager->getGui()->registerCallback("fovVignette.setColor",
      "Value to adjust the color of the vignette.", std::function([this](std::string value) {
        mPluginSettings->mVignetteSettings.mColor = static_cast<std::string>(value);
      }));
  mPluginSettings->mVignetteSettings.mColor.connectAndTouch([this](std::string value) {
    mGuiManager->getGui()->callJavascript("CosmoScout.fovVignette.setColorValue", value);
  });

  // register callback for fov vignette lower velocity threshold
  mGuiManager->getGui()->registerCallback("fovVignette.setLowerThreshold",
      "Value to adjust the minimum velocity threshold when the vignette should be drawn (values "
      "from 0 to 10% of max. velocity).",
      std::function([this](double value) {
        mPluginSettings->mVignetteSettings.mLowerVelocityThreshold = static_cast<float>(value);
      }));
  mPluginSettings->mVignetteSettings.mLowerVelocityThreshold.connectAndTouch(
      [this](float value) { mGuiManager->setSliderValue("fovVignette.setLowerThreshold", value); });

  // register callback for fov vignette upper velocity threshold
  mGuiManager->getGui()->registerCallback("fovVignette.setUpperThreshold",
      "Value to adjust the maximum velocity threshold when the vignette should be set to the "
      "radius specified in the settings (values from 90 to 100% of max. velocity).",
      std::function([this](double value) {
        mPluginSettings->mVignetteSettings.mUpperVelocityThreshold = static_cast<float>(value);
      }));
  mPluginSettings->mVignetteSettings.mUpperVelocityThreshold.connectAndTouch(
      [this](float value) { mGuiManager->setSliderValue("fovVignette.setUpperThreshold", value); });

  // register callback for fov vignette fade duration slider
  mGuiManager->getGui()->registerCallback("fovVignette.setDuration",
      "Value to adjust the fade animation for the vignette (in seconds).",
      std::function(
          [this](double value) { mPluginSettings->mVignetteSettings.mFadeDuration = value; }));
  mPluginSettings->mVignetteSettings.mFadeDuration.connectAndTouch(
      [this](double value) { mGuiManager->setSliderValue("fovVignette.setDuration", value); });

  // register callback for fov vignette fade deadzone
  mGuiManager->getGui()->registerCallback("fovVignette.setDeadzone",
      "Value to adjust the deadzone wherein the vignette ignores short movements (in seconds).",
      std::function(
          [this](double value) { mPluginSettings->mVignetteSettings.mFadeDeadzone = value; }));
  mPluginSettings->mVignetteSettings.mFadeDeadzone.connectAndTouch(
      [this](double value) { mGuiManager->setSliderValue("fovVignette.setDeadzone", value); });

  // Load settings.
  onLoad();

  logger().info("Loading done.");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::deInit() {
  logger().info("Unloading plugin...");

  // remove settings tab
  mGuiManager->removeSettingsSection("VR Accessibility");
  // remove callbacks
  mGuiManager->getGui()->unregisterCallback("floorGrid.setEnabled");
  mGuiManager->getGui()->unregisterCallback("floorGrid.setSize");
  mGuiManager->getGui()->unregisterCallback("floorGrid.setOffset");
  mGuiManager->getGui()->unregisterCallback("floorGrid.setAlpha");
  mGuiManager->getGui()->unregisterCallback("floorGrid.setColor");
  mGuiManager->getGui()->unregisterCallback("fovVignette.setEnabled");
  mGuiManager->getGui()->unregisterCallback("fovVignette.setDebug");
  mGuiManager->getGui()->unregisterCallback("fovVignette.setEnableDynamicRadius");
  mGuiManager->getGui()->unregisterCallback("fovVignette.setEnableVerticalOnly");
  mGuiManager->getGui()->unregisterCallback("fovVignette.setInnerRadius");
  mGuiManager->getGui()->unregisterCallback("fovVignette.setOuterRadius");
  mGuiManager->getGui()->unregisterCallback("fovVignette.setColor");
  mGuiManager->getGui()->unregisterCallback("fovVignette.setLowerThreshold");
  mGuiManager->getGui()->unregisterCallback("fovVignette.setUpperThreshold");
  mGuiManager->getGui()->unregisterCallback("fovVignette.setDuration");
  mGuiManager->getGui()->unregisterCallback("fovVignette.setDeadzone");

  mAllSettings->onLoad().disconnect(mOnLoadConnection);
  mAllSettings->onSave().disconnect(mOnSaveConnection);

  logger().info("Unloading done.");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::update() {
  // on first update, reset color picker to original color from the settings json
  if (resetColorPicker) {
    // reread settings from json
    from_json(mAllSettings->mPlugins.at("csp-vr-accessibility"), *mPluginSettings);
    // reset grid color into picker
    mGuiManager->getGui()->callJavascript(
        "CosmoScout.floorGrid.setColorValue", mPluginSettings->mGridSettings.mColor.get());
    // reset vignette color into picker
    mGuiManager->getGui()->callJavascript(
        "CosmoScout.fovVignette.setColorValue", mPluginSettings->mVignetteSettings.mColor.get());
    // clear flag
    resetColorPicker = false;
  }

  mGrid->update();

  if (mPluginSettings->mVignetteSettings.mUseDynamicRadius.get()) {
    mVignette->updateDynamicRadiusVignette();
  } else {
    mVignette->updateFadeAnimatedVignette();
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::onLoad() {
  // Read settings from JSON.
  from_json(mAllSettings->mPlugins.at("csp-vr-accessibility"), *mPluginSettings);

  // Create & configure FloorGrid
  mGrid = std::make_shared<FloorGrid>(mSolarSystem, mPluginSettings->mGridSettings);
  mGrid->configure(mPluginSettings->mGridSettings);
  // Create & configure FovVignette
  mVignette = std::make_shared<FovVignette>(mSolarSystem, mPluginSettings->mVignetteSettings);
  mVignette->configure(mPluginSettings->mVignetteSettings);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

glm::vec4 Plugin::GetColorFromHexString(std::string color) {
  // cut off # symbol
  color = color.substr(1);
  // separate into colors
  std::string red{color.substr(0, 2)};
  std::string green{color.substr(2, 2)};
  std::string blue{color.substr(4, 2)};
  // translate to value and sort into vector
  glm::vec4 vector{static_cast<float>(std::stoul(red, nullptr, 16)) / 255,
      static_cast<float>(std::stoul(green, nullptr, 16)) / 255,
      static_cast<float>(std::stoul(blue, nullptr, 16)) / 255, 1.0F};
  return vector;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::vraccessibility
