////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "Plugin.hpp"
#include "logger.hpp"

// TODO: #include "UserStudy.hpp" neccessary?

#include <VistaKernel/VistaSystem.h>
#include <VistaKernel/GraphicsManager/VistaSceneGraph.h>

#include <iostream>

////////////////////////////////////////////////////////////////////////////////////////////////////

EXPORT_FN cs::core::PluginBase* create() {
  return new csp::userstudy::Plugin;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

EXPORT_FN void destroy(cs::core::PluginBase* pluginBase) {
  delete pluginBase; // NOLINT(cppcoreguidelines-owning-memory)
}

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace csp::userstudy {

////////////////////////////////////////////////////////////////////////////////////////////////////

// clang-format off

// NOLINTNEXTLINE
NLOHMANN_JSON_SERIALIZE_ENUM(Plugin::Settings::StageType, {
  {Plugin::Settings::StageType::eNone, nullptr},
  {Plugin::Settings::StageType::eCheckpoint, "checkpoint"},
  {Plugin::Settings::StageType::eRequestFMS, "requestFMS"},
  {Plugin::Settings::StageType::eSwitchScenario, "switchScenario"},
});

// clang-format on

////////////////////////////////////////////////////////////////////////////////////////////////////

void from_json(nlohmann::json const& j, Plugin::Settings::Stage& o) {
  cs::core::Settings::deserialize(j, "type", o.mType);
  cs::core::Settings::deserialize(j, "bookmark", o.mBookmark);
  cs::core::Settings::deserialize(j, "scale", o.mScaling);

  if (o.mType.get() == Plugin::Settings::StageType::eNone) {
    throw cs::core::Settings::DeserializationException(
      "'type'", "Invalid stage type given! Should be one of the types outlined in the README.md"
    );
  }
}

void to_json(nlohmann::json& j, Plugin::Settings::Stage const& o) {
  cs::core::Settings::serialize(j, "type", o.mType);
  cs::core::Settings::serialize(j, "bookmark", o.mBookmark);
  cs::core::Settings::serialize(j, "scale", o.mScaling);
}

void from_json(nlohmann::json const& j, Plugin::Settings::Scenario& o) {
  cs::core::Settings::deserialize(j, "name", o.mName);
  cs::core::Settings::deserialize(j, "path", o.mPath);
}

void to_json(nlohmann::json& j, Plugin::Settings::Scenario const& o) {
  cs::core::Settings::serialize(j, "name", o.mName);
  cs::core::Settings::serialize(j, "path", o.mPath);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void from_json(nlohmann::json const& j, Plugin::Settings& o) {
  cs::core::Settings::deserialize(j, "enabled", o.mEnabled);
  cs::core::Settings::deserialize(j, "debug", o.mDebug);
  cs::core::Settings::deserialize(j, "otherScenarios", o.mOtherScenarios);
  cs::core::Settings::deserialize(j, "stages", o.mStages);
}

void to_json(nlohmann::json& j, Plugin::Settings const& o) {
  cs::core::Settings::serialize(j, "enabled", o.mEnabled);
  cs::core::Settings::serialize(j, "debug", o.mDebug);
  cs::core::Settings::serialize(j, "otherScenarios", o.mOtherScenarios);
  cs::core::Settings::serialize(j, "stages", o.mStages);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool Plugin::Settings::Scenario::operator==(Plugin::Settings::Scenario const& other) const {
  return mName.get() == other.mName.get() && mPath.get() == other.mPath.get();
}

bool Plugin::Settings::Stage::operator==(Plugin::Settings::Stage const& other) const {
  return mType.get() == other.mType.get() && mBookmark.get() == other.mBookmark.get() && mScaling.get() == other.mScaling.get();
}

bool Plugin::Settings::operator==(Plugin::Settings const& other) const {
  return mOtherScenarios == other.mOtherScenarios && mStages == other.mStages;
}

bool Plugin::Settings::operator!=(Plugin::Settings const& other) const {
  return !(*this == other);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::init() {

  logger().info("Loading plugin...");

  mOnLoadConnection = mAllSettings->onLoad().connect([this]() { onLoad(); });
  mOnSaveConnection = mAllSettings->onSave().connect(
      [this]() { mAllSettings->mPlugins["csp-user-study"] = *mPluginSettings; });
  
  // TODO: Register Callbacks here if needed

  onLoad();

  logger().info("Loading done.");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::update() {
  if (mPluginSettings->mEnabled.get()) {
    // TODO: check active stage index & which stages to hide/unhide
    // TODO: check flythrough for checkpoints?
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::deInit() {
  logger().info("Unloading plugin...");

  // remove stages
  unload(*mPluginSettings);

  mAllSettings->onLoad().disconnect(mOnLoadConnection);
  mAllSettings->onSave().disconnect(mOnSaveConnection);

  logger().info("Unloading done.");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::onLoad() {
  // store current settings
  Plugin::Settings oldSettings = *mPluginSettings;

  // Read settings from JSON.
  from_json(mAllSettings->mPlugins.at("csp-user-study"), *mPluginSettings);

  // Check if settings changed
  if (*mPluginSettings != oldSettings) {

    // remove existing stages
    unload(oldSettings);

    // add stages
    VistaSceneGraph* pSG = GetVistaSystem()->GetGraphicsManager()->GetSceneGraph();
    for (auto const& stageSettings : mPluginSettings->mStages){
      
      Stage s;

      // find anchor for stage
      
    }
  }

}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::unload(Plugin::Settings pluginSettings) {

  // TODO: Remove stages

}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::userstudy
