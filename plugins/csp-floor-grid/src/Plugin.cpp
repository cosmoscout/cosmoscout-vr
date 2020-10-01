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
#include "logger.hpp"

////////////////////////////////////////////////////////////////////////////////////////////////////

EXPORT_FN cs::core::PluginBase* create() {
  return new csp::floorgrid::Plugin;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

EXPORT_FN void destroy(cs::core::PluginBase* pluginBase) {
  delete pluginBase; // NOLINT(cppcoreguidelines-owning-memory)
}

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace csp::floorgrid {

////////////////////////////////////////////////////////////////////////////////////////////////////

void from_json(nlohmann::json const& j, Plugin::Settings& o) {
  cs::core::Settings::deserialize(j, "enabled", o.mEnabled);
  cs::core::Settings::deserialize(j, "size", o.mSize);
  cs::core::Settings::deserialize(j, "offset", o.mOffset);
  cs::core::Settings::deserialize(j, "falloff", o.mFalloff);
  cs::core::Settings::deserialize(j, "texture", o.mTexture);
  cs::core::Settings::deserialize(j, "alpha", o.mAlpha);
  cs::core::Settings::deserialize(j, "color", o.mColor);
}

void to_json(nlohmann::json& j, Plugin::Settings const& o) {
  cs::core::Settings::serialize(j, "enabled", o.mEnabled);
  cs::core::Settings::serialize(j, "size", o.mSize);
  cs::core::Settings::serialize(j, "offset", o.mOffset);
  cs::core::Settings::serialize(j, "falloff", o.mFalloff);
  cs::core::Settings::serialize(j, "texture", o.mTexture);
  cs::core::Settings::serialize(j, "alpha", o.mAlpha);
  cs::core::Settings::serialize(j, "color", o.mColor);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::init() {
  logger().info("Loading plugin...");

  mOnLoadConnection = mAllSettings->onLoad().connect([this]() { onLoad(); });
  mOnSaveConnection = mAllSettings->onSave().connect(
      [this]() { mAllSettings->mPlugins["csp-floor-grid"] = *mPluginSettings; });

  // add settings to GUI
  mGuiManager->addSettingsSectionToSideBarFromHTML(
      "Floor Grid", "blur_circular", "../share/resources/gui/floor_grid_settings.html"
      );
  mGuiManager->addScriptToGuiFromJS(
      "../share/resources/gui/js/csp-floor-grid.js"
      );
  // register callback for enable checkbox
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
  // register callback for color picker
  mGuiManager->getGui()->registerCallback(
      "floorGrid.setColor",
      "Value to adjust color of the grid.",
      std::function([this](std::string value) { mPluginSettings->mColor = static_cast<std::string>(std::move(value)); })
      );
  mPluginSettings->mColor.connectAndTouch(
      [this](const std::string& value) {
        mGuiManager->getGui()->callJavascript("CosmoScout.gui.setTextboxValue", "floorGrid.setColor", false, value);
      });
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
  mGrid->update();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::onLoad() {
  //cs::core::GraphicsEngine::enableGLDebug();

  // Read settings from JSON.
  from_json(mAllSettings->mPlugins.at("csp-floor-grid"), *mPluginSettings);

  mGrid = std::make_shared<FloorGrid>(mSolarSystem);
  mGrid->configure(mPluginSettings);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::floorgrid
