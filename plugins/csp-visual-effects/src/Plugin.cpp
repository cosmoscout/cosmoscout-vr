////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "Plugin.hpp"

#include "../../../src/cs-core/GuiManager.hpp"
#include "SolarFlares.hpp"
#include "logger.hpp"

////////////////////////////////////////////////////////////////////////////////////////////////////

EXPORT_FN cs::core::PluginBase* create() {
  return new csp::visualeffects::Plugin;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

EXPORT_FN void destroy(cs::core::PluginBase* pluginBase) {
  delete pluginBase; // NOLINT(cppcoreguidelines-owning-memory)
}

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace csp::visualeffects {

////////////////////////////////////////////////////////////////////////////////////////////////////

void from_json(nlohmann::json const& j, Plugin::Settings& o) {
  cs::core::Settings::deserialize(j, "solarFlares", o.mSolarFlares);
}

void to_json(nlohmann::json& j, Plugin::Settings const& o) {
  cs::core::Settings::serialize(j, "solarFlares", o.mSolarFlares);
}

////////////////////////////////////////////////////////////////////////////////////////////////////


void from_json(nlohmann::json const& j, Plugin::Settings::SolarFlares& o) {
}

void to_json(nlohmann::json& j, Plugin::Settings::SolarFlares const& o) {
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::init() {
  logger().info("Loading plugin...");

  mOnLoadConnection = mAllSettings->onLoad().connect([this]() { onLoad(); });
  mOnSaveConnection = mAllSettings->onSave().connect(
      [this]() { mAllSettings->mPlugins["csp-visual-effects"] = *mPluginSettings; });

  onLoad();

  logger().info("Loading done.");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::update() {
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::deInit() {
  logger().info("Unloading plugin...");

  mGuiManager->removePluginTab("Visual Effects");
  mGuiManager->removeSettingsSection("Visual Effects");

  mAllSettings->onLoad().disconnect(mOnLoadConnection);
  mAllSettings->onSave().disconnect(mOnSaveConnection);

  logger().info("Unloading done.");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::onLoad() {
  // Read settings from JSON.
  from_json(mAllSettings->mPlugins.at("csp-visual-effects"), *mPluginSettings);

  for (auto const& settings : mPluginSettings->mSolarFlares) {
    auto solarFlares = std::make_shared<SolarFlares>(mPluginSettings, mSolarSystem);
    solarFlares->setParentName(settings.first);
    mSolarFlares.emplace(settings.first, solarFlares);
    logger().info("Loaded solar flares for {}.", settings.first);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::visualeffects
