////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "Plugin.hpp"

#include "../../../src/cs-core/InputManager.hpp"
#include "../../../src/cs-core/Settings.hpp"
#include "../../../src/cs-core/SolarSystem.hpp"
#include "../../../src/cs-utils/logger.hpp"
#include "../../../src/cs-utils/utils.hpp"
#include "SimpleBody.hpp"
#include "logger.hpp"

////////////////////////////////////////////////////////////////////////////////////////////////////

EXPORT_FN cs::core::PluginBase* create() {
  return new csp::simplebodies::Plugin;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

EXPORT_FN void destroy(cs::core::PluginBase* pluginBase) {
  delete pluginBase; // NOLINT(cppcoreguidelines-owning-memory)
}

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace csp::simplebodies {

////////////////////////////////////////////////////////////////////////////////////////////////////

void from_json(nlohmann::json const& j, Plugin::Settings::SimpleBody& o) {
  cs::core::Settings::deserialize(j, "texture", o.mTexture);
}

void to_json(nlohmann::json& j, Plugin::Settings::SimpleBody const& o) {
  cs::core::Settings::serialize(j, "texture", o.mTexture);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void from_json(nlohmann::json const& j, Plugin::Settings& o) {
  cs::core::Settings::deserialize(j, "bodies", o.mSimpleBodies);
}

void to_json(nlohmann::json& j, Plugin::Settings const& o) {
  cs::core::Settings::serialize(j, "bodies", o.mSimpleBodies);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::init() {

  logger().info("Loading plugin...");

  mOnLoadConnection = mAllSettings->onLoad().connect([this]() { onLoad(); });
  mOnSaveConnection = mAllSettings->onSave().connect(
      [this]() { mAllSettings->mPlugins["csp-simple-bodies"] = mPluginSettings; });

  // Load settings.
  onLoad();

  logger().info("Loading done.");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::deInit() {
  logger().info("Unloading plugin...");

  for (auto const& simpleBody : mSimpleBodies) {
    mSolarSystem->unregisterBody(simpleBody.second);
    mInputManager->unregisterSelectable(simpleBody.second);
  }

  mAllSettings->onLoad().disconnect(mOnLoadConnection);
  mAllSettings->onSave().disconnect(mOnSaveConnection);

  logger().info("Unloading done.");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::onLoad() {
  // Read settings from JSON.
  from_json(mAllSettings->mPlugins.at("csp-simple-bodies"), mPluginSettings);

  // First try to re-configure existing simpleBodies. We assume that they are similar if they have
  // the same name in the settings (which means they are attached to an anchor with the same name).
  auto simpleBody = mSimpleBodies.begin();
  while (simpleBody != mSimpleBodies.end()) {
    auto settings = mPluginSettings.mSimpleBodies.find(simpleBody->first);
    if (settings != mPluginSettings.mSimpleBodies.end()) {
      // If there are settings for this simpleBody, reconfigure it.
      mAllSettings->initAnchor(*simpleBody->second, settings->first);
      simpleBody->second->configure(settings->second);

      ++simpleBody;
    } else {
      // Else delete it.
      mSolarSystem->unregisterBody(simpleBody->second);
      mInputManager->unregisterSelectable(simpleBody->second);
      simpleBody = mSimpleBodies.erase(simpleBody);
    }
  }

  // Then add new simpleBodies.
  for (auto const& settings : mPluginSettings.mSimpleBodies) {
    if (mSimpleBodies.find(settings.first) != mSimpleBodies.end()) {
      continue;
    }

    auto simpleBody = std::make_shared<SimpleBody>(mAllSettings, mSolarSystem, settings.first);
    simpleBody->configure(settings.second);
    simpleBody->setSun(mSolarSystem->getSun());

    mSolarSystem->registerBody(simpleBody);
    mInputManager->registerSelectable(simpleBody);

    mSimpleBodies.emplace(settings.first, simpleBody);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::simplebodies
