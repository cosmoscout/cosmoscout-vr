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

  mPluginSettings = mAllSettings->mPlugins.at("csp-satellites");

  for (auto const& settings : mPluginSettings.mSatellites) {
    auto anchor = mAllSettings->mAnchors.find(settings.first);

    if (anchor == mAllSettings->mAnchors.end()) {
      throw std::runtime_error(
          "There is no Anchor \"" + settings.first + "\" defined in the settings.");
    }

    auto satellite = std::make_shared<Satellite>(
        settings.second, settings.first, mSceneGraph, mAllSettings, mSolarSystem);

    satellite->setSun(mSolarSystem->getSun());
    mSolarSystem->registerBody(satellite);

    mSatellites.push_back(satellite);
  }

  logger().info("Loading done.");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::deInit() {
  logger().info("Unloading plugin...");

  for (auto const& satellite : mSatellites) {
    mSolarSystem->unregisterBody(satellite);
  }

  logger().info("Unloading done.");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::satellites
