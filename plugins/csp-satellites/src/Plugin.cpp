////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "Plugin.hpp"

#include "Satellite.hpp"
#include "logger.hpp"

#include "../../../src/cs-core/SolarSystem.hpp"
#include "../../../src/cs-utils/logger.hpp"

////////////////////////////////////////////////////////////////////////////////////////////////////

EXPORT_FN cs::core::PluginBase* create() {
  return new csp::satellites::Plugin;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

EXPORT_FN void destroy(cs::core::PluginBase* pluginBase) {
  delete pluginBase; // NOLINT(cppcoreguidelines-owning-memory)
}

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace csp::satellites {

////////////////////////////////////////////////////////////////////////////////////////////////////

void from_json(nlohmann::json const& j, Plugin::Settings::Satellite& o) {
  cs::core::Settings::deserialize(j, "modelFile", o.mModelFile);
  cs::core::Settings::deserialize(j, "environmentMap", o.mEnvironmentMap);
}

void to_json(nlohmann::json& j, Plugin::Settings::Satellite const& o) {
  cs::core::Settings::serialize(j, "modelFile", o.mModelFile);
  cs::core::Settings::serialize(j, "environmentMap", o.mEnvironmentMap);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void from_json(nlohmann::json const& j, Plugin::Settings& o) {
  cs::core::Settings::deserialize(j, "satellites", o.mSatellites);
}

void to_json(nlohmann::json& j, Plugin::Settings const& o) {
  cs::core::Settings::serialize(j, "satellites", o.mSatellites);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::init() {

  logger().info("Loading plugin...");

  mOnLoadConnection = mAllSettings->onLoad().connect([this]() { onLoad(); });
  mOnSaveConnection = mAllSettings->onSave().connect([this]() { onSave(); });

  // Load settings.
  onLoad();

  logger().info("Loading done.");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::deInit() {
  logger().info("Unloading plugin...");

  // Save settings as this plugin may get reloaded.
  onSave();

  mSatellites.clear();

  mAllSettings->onLoad().disconnect(mOnLoadConnection);
  mAllSettings->onSave().disconnect(mOnSaveConnection);

  logger().info("Unloading done.");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::update() {
  for (auto const& satellite : mSatellites) {
    satellite->update();
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::onLoad() {

  mSatellites.clear();
  
  // Read settings from JSON.
  mPluginSettings = mAllSettings->mPlugins.at("csp-satellites");

  for (auto const& settings : mPluginSettings.mSatellites) {
    mSatellites.push_back(std::make_shared<Satellite>(
        settings.second, settings.first, mSceneGraph, mAllSettings, mSolarSystem));
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::onSave() {
  mAllSettings->mPlugins["csp-satellites"] = mPluginSettings;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::satellites
