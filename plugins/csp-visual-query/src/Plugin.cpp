////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "Plugin.hpp"

#include "../../../src/cs-core/GuiManager.hpp"
#include "logger.hpp"

#include "../../../src/cs-utils/filesystem.hpp"
#include "common-nodes/Int/Int.hpp"
#include "common-nodes/Real/Real.hpp"
#include "common-nodes/RealVec2/RealVec2.hpp"
#include "common-nodes/RealVec4/RealVec4.hpp"
#include "common-nodes/TimeInterval/TimeInterval.hpp"
#include "operation-nodes/DifferenceImage2D/DifferenceImage2D.hpp"
#include "operation-nodes/TransferFunction/TransferFunction.hpp"
#include "output-nodes/CoverageInfo/CoverageInfo.hpp"
#include "output-nodes/OverlayRenderer/OverlayRender.hpp"
#include "output-nodes/VolumeRenderer/VolumeRenderer.hpp"
#include "source-nodes/JsonVolumeFileLoader/JsonVolumeFileLoader.hpp"
#include "source-nodes/RandomDataSource2D/RandomDataSource2D.hpp"
#include "source-nodes/RandomDataSource3D/RandomDataSource3D.hpp"
#include "source-nodes/WCSCoverage/WCSCoverage.hpp"
#include "source-nodes/WCSImage2D/WCSImage2D.hpp"

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

  onLoad();

  // Restart the node editor if the port changes.
  mPluginSettings.mPort.connectAndTouch([this](uint16_t port) { setupNodeEditor(port); });

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

  // load WCS
  mPluginSettings.mWebCoverages.clear();
  for (std::string const& url : mPluginSettings.mWcsUrl) {
    mPluginSettings.mWebCoverages.emplace_back(
        url, csl::ogc::WebServiceBase::CacheMode::eAlways, "wcs-cache");
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

  factory.registerSocketType("Coverage", "#8e38ff");
  factory.registerSocketType("Image2D", "#3333ff");
  factory.registerSocketType("Volume3D", "#ff3333");
  factory.registerSocketType("WCSTime", "#b08ab3");
  factory.registerSocketType("WCSBounds", "#b08ab3");

  factory.registerSocketType("Real", "#b2e2e2");
  factory.registerSocketType("RVec2", "#66c2a4");
  factory.registerSocketType("RVec3", "#2ca25f");
  factory.registerSocketType("RVec4", "#006d2c");

  factory.registerSocketType("Int", "#fecc5c");
  factory.registerSocketType("IVec2", "#fd8d3c");
  factory.registerSocketType("IVec3", "#f03b20");
  factory.registerSocketType("IVec4", "#bd0026");

  factory.registerSocketType("Coverage", "#8e38ff");
  factory.registerSocketType("Image2D", "#3333ff");
  factory.registerSocketType("Volume3D", "#ff3333");
  factory.registerSocketType("WCSTimeIntervals", "#FFB319");
  factory.registerSocketType("WCSTime", "#b08ab3");
  factory.registerSocketType("WCSBounds", "#b08ab3");
  factory.registerSocketType("LUT",
      "linear-gradient(90deg, rgba(255,0,0,1) 0%, rgba(255,154,0,1) 10%, rgba(208,222,33,1) "
      "20%, rgba(79,220,74,1) 30%, rgba(63,218,216,1) 40%, rgba(47,201,226,1) 50%, "
      "rgba(28,127,238,1) 60%, rgba(95,21,242,1) 70%, rgba(186,12,248,1) 80%, "
      "rgba(251,7,217,1) 90%, rgba(255,0,0,1) 100%);");

  factory.registerLibrary(
      R"HTML(<script type="module" src="third-party/js/transfer-function-editor.js"></script>)HTML");

  // Register control types:
  factory.registerControlType(cs::utils::filesystem::loadToString(
      "../share/resources/nodes/csp-visual-query/DropDownControl.js"));
  factory.registerControlType(cs::utils::filesystem::loadToString(
      "../share/resources/nodes/csp-visual-query/RealInputControl.js"));

  // Now, we register our custom node types. Any parameter given to this method, will later be
  // passed to the constructor of the node instances. For more information, see the documentation of
  // NodeFactory::registerNodeType().

  // Commons
  factory.registerNodeType<Int>();
  factory.registerNodeType<Real>();
  factory.registerNodeType<RealVec2>();
  factory.registerNodeType<RealVec4>();

  // Operations
  factory.registerNodeType<DifferenceImage2D>();
  factory.registerNodeType<TimeInterval>(mTimeControl);
  factory.registerNodeType<TransferFunction>();

  // Outputs
  factory.registerNodeType<CoverageInfo>();
  factory.registerNodeType<OverlayRender>(mSolarSystem, mAllSettings);
  factory.registerNodeType<VolumeRenderer>(mSolarSystem, mAllSettings);

  // Sources
  factory.registerNodeType<WCSCoverage>(mPluginSettings.mWebCoverages);
  factory.registerNodeType<WCSImage2D>();
  factory.registerNodeType<JsonVolumeFileLoader>();
  factory.registerNodeType<RandomDataSource2D>();
  factory.registerNodeType<RandomDataSource3D>();

  // Finally, create the node editor. It will start the server so that we can now open a web browser
  // and navigate to localhost:<port>.
  mNodeEditor = std::make_unique<csl::nodeeditor::NodeEditor>(port, factory);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::visualquery
