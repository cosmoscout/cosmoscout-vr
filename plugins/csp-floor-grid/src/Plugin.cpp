////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "Plugin.hpp"

#include "../../../src/cs-core/Settings.hpp"
#include "../../../src/cs-core/SolarSystem.hpp"
#include "../../../src/cs-utils/logger.hpp"
#include "../../../src/cs-utils/utils.hpp"
#include "../../../src/cs-core/GraphicsEngine.hpp"
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
}

void to_json(nlohmann::json& j, Plugin::Settings const& o) {
  cs::core::Settings::serialize(j, "enabled", o.mEnabled);
  cs::core::Settings::serialize(j, "size", o.mSize);
  cs::core::Settings::serialize(j, "offset", o.mOffset);
  cs::core::Settings::serialize(j, "falloff", o.mFalloff);
  cs::core::Settings::serialize(j, "texture", o.mTexture);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::init() {

  logger().info("Loading plugin...");

  mOnLoadConnection = mAllSettings->onLoad().connect([this]() { onLoad(); });
  mOnSaveConnection = mAllSettings->onSave().connect(
      [this]() { mAllSettings->mPlugins["csp-floor-grid"] = *mPluginSettings; });

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

void Plugin::onLoad() {
  cs::core::GraphicsEngine::enableGLDebug();
  // Read settings from JSON.
  from_json(mAllSettings->mPlugins.at("csp-floor-grid"), *mPluginSettings);

  auto grid = std::make_shared<FloorGrid>(mSolarSystem);
  grid->configure(Plugin::Settings({
      mPluginSettings->mEnabled,
      mPluginSettings->mSize,
      mPluginSettings->mOffset,
      mPluginSettings->mFalloff,
      mPluginSettings->mTexture
  }));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::floorgrid
