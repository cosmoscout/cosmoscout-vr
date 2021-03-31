////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "Plugin.hpp"
#include "logger.hpp"

#include "../../../src/cs-core/GraphicsEngine.hpp"
#include "../../../src/cs-core/GuiManager.hpp"
#include "../../../src/cs-core/InputManager.hpp"
#include "../../../src/cs-core/SolarSystem.hpp"
#include "../../../src/cs-core/TimeControl.hpp"
#include "../../../src/cs-scene/CelestialAnchorNode.hpp"
#include "../../../src/cs-utils/convert.hpp"

////////////////////////////////////////////////////////////////////////////////////////////////////

EXPORT_FN cs::core::PluginBase* create() {
  return new csp::guide::Plugin;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

EXPORT_FN void destroy(cs::core::PluginBase* pluginBase) {
  delete pluginBase;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace csp::guide {

////////////////////////////////////////////////////////////////////////////////////////////////////

void from_json(nlohmann::json const& j, Plugin::Settings& o) {
}

void to_json(nlohmann::json& j, Plugin::Settings const& o) {
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::init() {

  logger().info("Loading plugin...");

  // Call onLoad whenever the settings are reloaded.
  mOnLoadConnection = mAllSettings->onLoad().connect([this]() { onLoad(); });

  // Store the current settings on save.
  mOnSaveConnection = mAllSettings->onSave().connect(
      [this]() { mAllSettings->mPlugins["csp-guide"] = mPluginSettings; });

  mGuiManager->addCssToGui("third-party/css/clippy.css");
  mGuiManager->addScriptToGuiFromJS("../share/resources/gui/js/csp-guide.js");

  // Load initial settings.
  onLoad();

  logger().info("Loading done.");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::deInit() {

  // Remove all items.
  unload(mPluginSettings);

  mAllSettings->onLoad().disconnect(mOnLoadConnection);
  mAllSettings->onSave().disconnect(mOnSaveConnection);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::onLoad() {
  unload(mPluginSettings);

  // Read settings from JSON.
  // from_json(mAllSettings->mPlugins.at("csp-guide"), mPluginSettings);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::unload(Settings const& pluginSettings) {
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::guide
