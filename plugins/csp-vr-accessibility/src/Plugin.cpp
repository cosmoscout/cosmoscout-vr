////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "Plugin.hpp"

#include <utility>

#include "../../../src/cs-core/Settings.hpp"
#include "../../../src/cs-core/SolarSystem.hpp"
#include "../../../src/cs-utils/logger.hpp"
#include "../../../src/cs-utils/utils.hpp"
#include "../../../src/cs-core/GraphicsEngine.hpp"
#include "../../../src/cs-core/GuiManager.hpp"
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

void from_json(nlohmann::json const& j, Plugin::Settings& o) {
  cs::core::Settings::deserialize(j, "gridEnabled", o.mEnabled);
  cs::core::Settings::deserialize(j, "gridSize",    o.mSize);
  cs::core::Settings::deserialize(j, "gridOffset",  o.mOffset);
  cs::core::Settings::deserialize(j, "gridFalloff", o.mFalloff);
  cs::core::Settings::deserialize(j, "gridTexture", o.mTexture);
  cs::core::Settings::deserialize(j, "gridAlpha",   o.mAlpha);
  cs::core::Settings::deserialize(j, "gridColor",   o.mColor);
  cs::core::Settings::deserialize(j, "vignetteEnabled",                o.mFovVignetteEnabled);
  cs::core::Settings::deserialize(j, "vignetteDebug",                  o.mFovVignetteDebug);
  cs::core::Settings::deserialize(j, "vignetteInnerRadius",            o.mFovVignetteInnerRadius);
  cs::core::Settings::deserialize(j, "vignetteOuterRadius",            o.mFovVignetteOuterRadius);
  cs::core::Settings::deserialize(j, "vignetteColor",                  o.mFovVignetteColor);
  cs::core::Settings::deserialize(j, "vignetteFadeDuration",           o.mFovVignetteFadeDuration);
  cs::core::Settings::deserialize(j, "vignetteFadeDeadzone",           o.mFovVignetteFadeDeadzone);
  cs::core::Settings::deserialize(j, "vignetteLowerVelocityThreshold", o.mFovVignetteLowerVelocityThreshold);
  cs::core::Settings::deserialize(j, "vignetteUpperVelocityThreshold", o.mFovVignetteUpperVelocityThreshold);
  cs::core::Settings::deserialize(j, "vignetteUseDynamicRadius",       o.mFovVignetteUseDynamicRadius);
}

void to_json(nlohmann::json& j, Plugin::Settings const& o) {
  cs::core::Settings::serialize(j, "gridEnabled", o.mEnabled);
  cs::core::Settings::serialize(j, "gridSize",    o.mSize);
  cs::core::Settings::serialize(j, "gridOffset",  o.mOffset);
  cs::core::Settings::serialize(j, "gridFalloff", o.mFalloff);
  cs::core::Settings::serialize(j, "gridTexture", o.mTexture);
  cs::core::Settings::serialize(j, "gridAlpha",   o.mAlpha);
  cs::core::Settings::serialize(j, "gridColor",   o.mColor);
  cs::core::Settings::serialize(j, "vignetteEnabled",                o.mFovVignetteEnabled);
  cs::core::Settings::serialize(j, "vignetteDebug",                  o.mFovVignetteDebug);
  cs::core::Settings::serialize(j, "vignetteInnerRadius",            o.mFovVignetteInnerRadius);
  cs::core::Settings::serialize(j, "vignetteOuterRadius",            o.mFovVignetteOuterRadius);
  cs::core::Settings::serialize(j, "vignetteColor",                  o.mFovVignetteColor);
  cs::core::Settings::serialize(j, "vignetteFadeDuration",           o.mFovVignetteFadeDuration);
  cs::core::Settings::serialize(j, "vignetteFadeDeadzone",           o.mFovVignetteFadeDeadzone);
  cs::core::Settings::serialize(j, "vignetteLowerVelocityThreshold", o.mFovVignetteLowerVelocityThreshold);
  cs::core::Settings::serialize(j, "vignetteUpperVelocityThreshold", o.mFovVignetteUpperVelocityThreshold);
  cs::core::Settings::serialize(j, "vignetteUseDynamicRadius",       o.mFovVignetteUseDynamicRadius);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::init() {
  logger().info("Loading plugin...");

  mOnLoadConnection = mAllSettings->onLoad().connect([this]() { onLoad(); });
  mOnSaveConnection = mAllSettings->onSave().connect(
      [this]() { mAllSettings->mPlugins["csp-vr-accessibility"] = *mPluginSettings; });

  // add settings to GUI
  mGuiManager->addSettingsSectionToSideBarFromHTML(
      "VR Accessibility", "blur_circular", "../share/resources/gui/vr_accessibility_settings.html"
      );
  mGuiManager->addScriptToGuiFromJS(
      "../share/resources/gui/js/csp-vr-accessibility.js"
      );
  // register callback for grid enable grid checkbox
  mGuiManager->getGui()->registerCallback(
      "floorGrid.setEnabled",
      "Enables or disables rendering the grid.",
      std::function([this](bool enable) { mPluginSettings->mEnabled = enable; })
      );
  mPluginSettings->mEnabled.connectAndTouch(
      [this](bool enable) { mGuiManager->setCheckboxValue("floorGrid.setEnabled", enable); }
      );
  // register callback for grid size slider
  mGuiManager->getGui()->registerCallback(
      "floorGrid.setSize",
      "Value scales the grid size between 0.5 (doubles the square size) and 2 (halves square size).",
      std::function([this](double value) { mPluginSettings->mSize = static_cast<float>(std::pow(2, value)); })
      );
  mPluginSettings->mSize.connectAndTouch(
      [this](float value) { mGuiManager->setSliderValue("floorGrid.setSize", std::round(std::log2(value))); }
      );
  // register callback for grid offset slider
  mGuiManager->getGui()->registerCallback(
      "floorGrid.setOffset",
      "Value to adjust downward offset of the grid.",
      std::function([this](double value) { mPluginSettings->mOffset = static_cast<float>(value); })
      );
  mPluginSettings->mOffset.connectAndTouch(
      [this](float value) { mGuiManager->setSliderValue("floorGrid.setOffset", value); }
      );
  // register callback for grid alpha slider
  mGuiManager->getGui()->registerCallback(
      "floorGrid.setAlpha",
      "Value to adjust grid opacity.",
      std::function([this](double value) { mPluginSettings->mAlpha = static_cast<float>(value); })
      );
  mPluginSettings->mAlpha.connectAndTouch(
      [this](float value) { mGuiManager->setSliderValue("floorGrid.setAlpha", value); }
      );
  // register callback for grid color picker
  mGuiManager->getGui()->registerCallback(
      "floorGrid.setColor",
      "Value to adjust color of the grid.",
      std::function([this](std::string value) { mPluginSettings->mColor = static_cast<std::string>(value); })
      );
  mPluginSettings->mColor.connectAndTouch(
      [this](std::string value) {
        mGuiManager->getGui()->callJavascript("CosmoScout.floorGrid.setColorValue", value);
      });
  // register callback for fov vignette enable checkbox
  mGuiManager->getGui()->registerCallback(
      "fovVignette.setEnabled",
      "Enables or disables a Vignette limiting the FoV on movement.",
      std::function([this](bool enable) { mPluginSettings->mFovVignetteEnabled = enable; })
      );
  mPluginSettings->mFovVignetteEnabled.connectAndTouch(
      [this](bool enable) { mGuiManager->setCheckboxValue("fovVignette.setEnabled", enable); }
  );
  // register callback for fov vignette debug checkbox
  mGuiManager->getGui()->registerCallback(
      "fovVignette.setDebug",
      "Enables or disables the vignette to be drawn permanently.",
      std::function([this](bool enable) { mPluginSettings->mFovVignetteDebug = enable; })
      );
  mPluginSettings->mFovVignetteDebug.connectAndTouch(
      [this](bool enable) { mGuiManager->setCheckboxValue("fovVignette.setDebug", enable); }
      );
  // register callback for fov vignette use dynamic radius
  mGuiManager->getGui()->registerCallback(
      "fovVignette.setEnableDynamicRadius",
      "Dynamically adjust the radius based on velocity instead of fading the vignette in.",
      std::function([this](bool enable) { mPluginSettings->mFovVignetteUseDynamicRadius = enable; })
      );
  mPluginSettings->mFovVignetteUseDynamicRadius.connectAndTouch(
      [this](bool enable) { mGuiManager->setCheckboxValue("fovVignette.setEnableDynamicRadius", enable); }
      );
  // register callback for fov vignette inner radius slider
  mGuiManager->getGui()->registerCallback(
      "fovVignette.setInnerRadius",
      "Value to adjust the inner radius (start of gradient) of the vignette.",
      std::function([this](double value) { mPluginSettings->mFovVignetteInnerRadius = static_cast<float>(value); })
      );
  mPluginSettings->mFovVignetteInnerRadius.connectAndTouch(
      [this](float value) { mGuiManager->setSliderValue("fovVignette.setInnerRadius", value); }
      );
  // register callback for fov vignette outer radius slider
  mGuiManager->getGui()->registerCallback(
      "fovVignette.setOuterRadius",
      "Value to adjust the outer radius (end of gradient) of the vignette.",
      std::function([this](double value) { mPluginSettings->mFovVignetteOuterRadius = static_cast<float>(value); })
      );
  mPluginSettings->mFovVignetteOuterRadius.connectAndTouch(
      [this](float value) { mGuiManager->setSliderValue("fovVignette.setOuterRadius", value); }
      );
  // register callback for fov vignette color picker
  mGuiManager->getGui()->registerCallback(
      "fovVignette.setColor",
      "Value to adjust the color of the vignette.",
      std::function([this](std::string value) { mPluginSettings->mFovVignetteColor = static_cast<std::string>(value); })
      );
  mPluginSettings->mFovVignetteColor.connectAndTouch(
      [this](std::string value) {
        mGuiManager->getGui()->callJavascript("CosmoScout.fovVignette.setColorValue", value);
      });
  // register callback for fov vignette lower velocity threshold
  mGuiManager->getGui()->registerCallback(
      "fovVignette.setLowerThreshold",
      "Value to adjust the minimum velocity threshold when the vignette should be drawn (values from 0 to 10% of max. velocity).",
      std::function([this] (double value) { mPluginSettings->mFovVignetteLowerVelocityThreshold = static_cast<float>(value); })
      );
  mPluginSettings->mFovVignetteLowerVelocityThreshold.connectAndTouch(
      [this](float value) { mGuiManager->setSliderValue("fovVignette.setLowerThreshold", value); }
      );
  // register callback for fov vignette upper velocity threshold
  mGuiManager->getGui()->registerCallback(
      "fovVignette.setUpperThreshold",
      "Value to adjust the maximum velocity threshold when the vignette should be set to the radius specified in the settings (values from 90 to 100% of max. velocity).",
      std::function([this] (double value) { mPluginSettings->mFovVignetteUpperVelocityThreshold = static_cast<float>(value); })
      );
  mPluginSettings->mFovVignetteUpperVelocityThreshold.connectAndTouch(
      [this](float value) { mGuiManager->setSliderValue("fovVignette.setUpperThreshold", value); }
      );
  // register callback for fov vignette fade duration slider
  mGuiManager->getGui()->registerCallback(
      "fovVignette.setDuration",
      "Value to adjust the fade animation for the vignette (in seconds).",
      std::function([this](double value) { mPluginSettings->mFovVignetteFadeDuration = value; })
      );
  mPluginSettings->mFovVignetteFadeDuration.connectAndTouch(
      [this](double value) { mGuiManager->setSliderValue("fovVignette.setDuration", value); }
      );
  // register callback for fov vignette fade deadzone
  mGuiManager->getGui()->registerCallback(
      "fovVignette.setDeadzone",
      "Value to adjust the deadzone wherein the vignett ignores short movements (in seconds).",
      std::function([this](double value) { mPluginSettings->mFovVignetteFadeDeadzone = value; })
      );
  mPluginSettings->mFovVignetteFadeDeadzone.connectAndTouch(
      [this](double value) { mGuiManager->setSliderValue("fovVignette.setDeadzone", value); }
      );
  
  // Load settings.
  onLoad();

  logger().info("Loading done.");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::deInit() {
  logger().info("Unloading plugin...");

  mAllSettings->onLoad().disconnect(mOnLoadConnection);
  mAllSettings->onSave().disconnect(mOnSaveConnection);

  logger().info("Unloading done.");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::update() {
  // on first update, reset color picker to original color from the settings json
  if(resetColorPicker){
    // reread grid color from json settings
    cs::core::Settings::deserialize(
        mAllSettings->mPlugins.at("csp-vr-accessibility"),
        "gridColor",
        mPluginSettings->mColor);
    // reread vignette color from json settings
    cs::core::Settings::deserialize(
        mAllSettings->mPlugins.at("csp-vr-accessibility"),
        "vignetteColor",
        mPluginSettings->mFovVignetteColor);
    // reset grid color into picker
    mGuiManager->getGui()->callJavascript("CosmoScout.floorGrid.setColorValue", mPluginSettings->mColor.get());
    //reset vignette color into picker
    mGuiManager->getGui()->callJavascript("CosmoScout.fovVignette.setColorValue", mPluginSettings->mFovVignetteColor.get());
    // clear flag
    resetColorPicker = false;
  }

  mGrid->update();

  if (mPluginSettings->mFovVignetteUseDynamicRadius.get()) {
    mVignette->updateDynamicRadiusVignette();
  }
  else {
    mVignette->updateFadeAnimatedVignette();
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::onLoad() {
  //cs::core::GraphicsEngine::enableGLDebug();

  // Read settings from JSON.
  from_json(mAllSettings->mPlugins.at("csp-vr-accessibility"), *mPluginSettings);

  // Create & configure FloorGrid
  mGrid = std::make_shared<FloorGrid>(mSolarSystem);
  mGrid->configure(mPluginSettings);
  // Create & configure FovVignette
  mVignette = std::make_shared<FovVignette>(mSolarSystem);
  mVignette->configure(mPluginSettings);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

glm::vec4 Plugin::GetColorFromHexString(std::string color) {
  // cut off # symbol
  color = color.substr(1);
  // separate into colors
  std::string red{color.substr(0,2)};
  std::string green{color.substr(2,2)};
  std::string blue{color.substr(4,2)};
  // translate to value and sort into vector
  glm::vec4 vector{
      static_cast<float>(std::stoul(red, nullptr, 16))/255,
      static_cast<float>(std::stoul(green, nullptr, 16))/255,
      static_cast<float>(std::stoul(blue, nullptr, 16))/255,
      1.0F};

  return vector;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::vraccessibility
