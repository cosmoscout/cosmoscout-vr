////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

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

void from_json(nlohmann::json const& j, Plugin::Settings::SimpleBody::Ring& o) {
  cs::core::Settings::deserialize(j, "texture", o.mTexture);
  cs::core::Settings::deserialize(j, "innerRadius", o.mInnerRadius);
  cs::core::Settings::deserialize(j, "outerRadius", o.mOuterRadius);
}

void to_json(nlohmann::json& j, Plugin::Settings::SimpleBody::Ring const& o) {
  cs::core::Settings::serialize(j, "texture", o.mTexture);
  cs::core::Settings::serialize(j, "innerRadius", o.mInnerRadius);
  cs::core::Settings::serialize(j, "outerRadius", o.mOuterRadius);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void from_json(nlohmann::json const& j, Plugin::Settings::SimpleBody& o) {
  cs::core::Settings::deserialize(j, "texture", o.mTexture);
  cs::core::Settings::deserialize(j, "albedo", o.mAlbedo);
  cs::core::Settings::deserialize(j, "primeMeridianInCenter", o.mPrimeMeridianInCenter);
  cs::core::Settings::deserialize(j, "ring", o.mRing);
}

void to_json(nlohmann::json& j, Plugin::Settings::SimpleBody const& o) {
  cs::core::Settings::serialize(j, "texture", o.mTexture);
  cs::core::Settings::serialize(j, "albedo", o.mAlbedo);
  cs::core::Settings::serialize(j, "primeMeridianInCenter", o.mPrimeMeridianInCenter);
  cs::core::Settings::serialize(j, "ring", o.mRing);
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

  for (auto const& [name, body] : mSimpleBodies) {
    unregisterBody(name);
  }

  mAllSettings->onLoad().disconnect(mOnLoadConnection);
  mAllSettings->onSave().disconnect(mOnSaveConnection);

  logger().info("Unloading done.");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::update() {
  for (auto const& body : mSimpleBodies) {
    body.second->update();
  }
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
    // If there are settings for this simpleBody, reconfigure it.
    if (settings != mPluginSettings.mSimpleBodies.end()) {
      simpleBody->second->setObjectName(settings->first);
      simpleBody->second->configure(settings->second);

      ++simpleBody;
    } else {
      // Else delete it.
      unregisterBody(simpleBody->first);
      simpleBody = mSimpleBodies.erase(simpleBody);
    }
  }

  // Then add new simpleBodies.
  for (auto const& settings : mPluginSettings.mSimpleBodies) {
    if (mSimpleBodies.find(settings.first) != mSimpleBodies.end()) {
      continue;
    }

    auto simpleBody = std::make_shared<SimpleBody>(mAllSettings, mSolarSystem);
    simpleBody->setObjectName(settings.first);
    simpleBody->configure(settings.second);

    auto object = mSolarSystem->getObject(settings.first);
    object->setSurface(simpleBody);
    object->setIntersectableObject(simpleBody);

    mSimpleBodies.emplace(settings.first, simpleBody);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::onSave() {
  mAllSettings->mPlugins["csp-simple-bodies"] = mPluginSettings;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::unregisterBody(std::string const& name) {
  auto object = mSolarSystem->getObject(name);
  object->setSurface(nullptr);
  object->setIntersectableObject(nullptr);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::simplebodies
