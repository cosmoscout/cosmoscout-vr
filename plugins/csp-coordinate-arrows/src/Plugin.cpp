////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "Plugin.hpp"

#include "Arrows.hpp"
#include "logger.hpp"

#include "../../../src/cs-core/GuiManager.hpp"

#include <VistaKernel/DisplayManager/VistaDisplayManager.h>
#include <VistaKernel/VistaSystem.h>

////////////////////////////////////////////////////////////////////////////////////////////////////

EXPORT_FN cs::core::PluginBase* create() {
  return new csp::coordinatearrows::Plugin;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

EXPORT_FN void destroy(cs::core::PluginBase* pluginBase) {
  delete pluginBase; // NOLINT(cppcoreguidelines-owning-memory)
}

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace csp::coordinatearrows {

////////////////////////////////////////////////////////////////////////////////////////////////////

void from_json(nlohmann::json const& j, Plugin::Settings& o) {
}

void to_json(nlohmann::json& j, Plugin::Settings const& o) {
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::init() {
  logger().info("Loading plugin...");

  mOnLoadConnection = mAllSettings->onLoad().connect([this]() { onLoad(); });
  mOnSaveConnection = mAllSettings->onSave().connect(
      [this]() { mAllSettings->mPlugins["csp-coordinate-arrows"] = *mPluginSettings; });

  mGuiManager->executeJavascriptFile("../share/resources/gui/js/csp-coordinate-arrows.js");
  mGuiManager->addSettingsSectionToSideBarFromHTML(
      "Coordinate Arrows", "label_off", "../share/resources/gui/coordinate-arrows-settings.html");

  mGuiManager->getGui()->registerCallback("coordinateArrows.enableArrows",
    "Enables or disables the rendering of the arrows.",
    std::function([this](bool value) {
      logger().info("Toggled enable button (WIP).");
      mPluginSettings->mEnableArrows = value;
    }));
  mPluginSettings->mEnableArrows.connectAndTouch([this](bool value) {
    mGuiManager->setCheckboxValue("coordinateArrows.enableArrows", true);
  });

  // Load settings
  onLoad();

  logger().info("Loading done.");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::update() {
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::deInit() {
  logger().info("Unloading plugin...");

  mGuiManager->removePluginTab("Coordinate Arrows");
  mGuiManager->removeSettingsSection("Coordinate Arrows");

  mAllSettings->onLoad().disconnect(mOnLoadConnection);
  mAllSettings->onSave().disconnect(mOnSaveConnection);

  logger().info("Unloading done.");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::onLoad() {
  // Read settings from JSON.
  from_json(mAllSettings->mPlugins.at("csp-coordinate-arrows"), *mPluginSettings);
  auto arrows = std::make_shared<Arrows>(mPluginSettings, mSolarSystem);
  arrows->setEnabled(true);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::coordinatearrows
