////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "Plugin.hpp"

#include "../../../src/cs-core/GuiManager.hpp"
#include "logger.hpp"

#include "../../../src/cs-core/Settings.hpp"

#include "../../csl-ogc/src/wcs/WebCoverageService.hpp"
#include "../../csl-ogc/src/wcs/WebCoverage.hpp"

#include "logger.hpp"
#include "outputNodes/Render.hpp"
#include "sourceNodes/WCSSource.hpp"
#include "sourceNodes/WCSImageLoader.hpp"
#include "commonNodes/NumberNode.hpp"

#include <vector>

////////////////////////////////////////////////////////////////////////////////////////////////////

EXPORT_FN cs::core::PluginBase* create() {
  return new csp::visualquery::Plugin;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

EXPORT_FN void destroy(cs::core::PluginBase* pluginBase) {
  delete pluginBase; // NOLINT(cppcoreguidelines-owning-memory)
}

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace csp::visualquery {

////////////////////////////////////////////////////////////////////////////////////////////////////

void from_json(nlohmann::json const& j, Plugin::Settings& o) {
  cs::core::Settings::deserialize(j, "port", o.mPort);
  cs::core::Settings::deserialize(j, "graph", o.mGraph);
  cs::core::Settings::deserialize(j, "wcs", o.mWcsUrl);
}

void to_json(nlohmann::json& j, Plugin::Settings const& o) {
  cs::core::Settings::serialize(j, "port", o.mPort);
  cs::core::Settings::serialize(j, "graph", o.mGraph);
}
////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::init() {
  logger().info("Loading plugin...");

  mOnLoadConnection = mAllSettings->onLoad().connect([this]() { onLoad(); });
  mOnSaveConnection = mAllSettings->onSave().connect([this]() { onSave(); });

  // Restart the node editor if the port changes.
  mPluginSettings.mPort.connect([this](uint16_t port) { setupNodeEditor(port); });

  onLoad();

  // load WCS 
  for (std::string url : mPluginSettings.mWcsUrl) {
    mPluginSettings.mWebCoverages.push_back(csl::ogc::WebCoverageService(
      url, csl::ogc::WebServiceBase::CacheMode::eAlways,
      "../../install/windows-Release/share/csp-visual-query/wcs-cache"
      )
    );
  }

  logger().info("Loading done.");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::deInit() {
  logger().info("Unloading plugin...");

  // Save settings as this plugin may get reloaded.
  onSave();

  mAllSettings->onLoad().disconnect(mOnLoadConnection);
  mAllSettings->onSave().disconnect(mOnSaveConnection);

  // Explicitly destroy the node editor so that we get any error messages before the "Unloading
  // done." message is printed.
  mNodeEditor.reset();

  logger().info("Unloading done.");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::update() {
  mNodeEditor->update();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::onLoad() {
  // Read settings from JSON.
  from_json(mAllSettings->mPlugins.at("csp-visual-query"), mPluginSettings);
  // from_json(mAllSettings->mPlugins.at("csp-wcs-overlays"), mPluginSettings);
  // If there is a graph defined in the settings, we give this to the node editor.
  if (mPluginSettings.mGraph.has_value()) {
    try {
      mNodeEditor->fromJSON(mPluginSettings.mGraph.value());
    } catch (std::exception const& e) { logger().warn("Failed to load node graph: {}", e.what()); }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::onSave() {

  // Save the current node graph layout.
  mPluginSettings.mGraph = mNodeEditor->toJSON();

  mAllSettings->mPlugins["csp-visual-query"] = mPluginSettings;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::setupNodeEditor(uint16_t port) {

  // Creating a node editor requires a node factory. This will be responsible for creating nodes
  // based on their names.
  csl::nodeeditor::NodeFactory factory;

  // First, we register the available socket types. For now, this only requires a unique name and a
  // color which will be used by the sockets. In this simple example, we only have number sockets.
  // The name of the socket will be used by the custom nodes when defining their inputs and outputs.
  // factory.registerSocketType("Number Value", "#b08ab3");
  
  factory.registerSocketType("GreyScaleGeoTexture", "#ffff00");
  factory.registerSocketType("WCSScalarField", "#b08ab3");

  factory.registerSocketType("WCSImage", "#b08ab3");
  factory.registerSocketType("WCSMinMax", "#b08ab3");
  factory.registerSocketType("WCSResolution", "#b08ab3");
  factory.registerSocketType("WCSTime", "#b08ab3");
  factory.registerSocketType("WCSBound", "#b08ab3");
  factory.registerSocketType("number Value", "#FFD480");

  // Now, we register our custom node types. Any parameter given to this method, will later be
  // passed to the constructor of the node instances. For more information, see the documentation of
  // NodeFactory::registerNodeType().
  factory.registerNodeType<WCSSource>();
  factory.registerNodeType<WCSImageLoader>(
    std::shared_ptr<std::vector<csl::ogc::WebCoverageService>>(&mPluginSettings.mWebCoverages));
  factory.registerNodeType<Render>();
  factory.registerNodeType<NumberNode>();

  // Finally, create the node editor. It will start the server so that we can now open a web browser
  // and navigate to localhost:<port>.
  mNodeEditor = std::make_unique<csl::nodeeditor::NodeEditor>(port, factory);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::visualquery
