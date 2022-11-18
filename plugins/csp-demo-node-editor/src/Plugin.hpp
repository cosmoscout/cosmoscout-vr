////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSP_DEMO_NODE_EDITOR_PLUGIN_HPP
#define CSP_DEMO_NODE_EDITOR_PLUGIN_HPP

#include "../../../src/cs-core/PluginBase.hpp"
#include "../../../src/cs-utils/Property.hpp"
#include "../../csl-node-editor/src/NodeEditor.hpp"

namespace csp::demonodeeditor {

/// This plugin serves as a demonstrator for the csl-node-editor plugin library. It shows how the
/// node editor can be used inside a plugin.
/// The plugin class maintains an instance of the csl::nodeeditor::NodeEditor class. It registers a
/// couple of custom node types with which some basic math operations can be modelled with the node
/// graph.
class Plugin : public cs::core::PluginBase {
 public:
  struct Settings {
    /// The port where the server should listen on. For example 9999.
    cs::utils::Property<uint16_t> mPort;

    /// Here we will store the current node graph if the scene gets saved. You could also save node
    /// graphs to separate json files instead and load them at runtime. The JSON format of this
    /// graph is defined in csl-node-editor/src/NodeEditor.hpp.
    std::optional<nlohmann::json> mGraph;
  };

  /// The plugin uses the standard plugin life cycle. On init, the settings are loaded and the node
  /// editor is set up. On update, the node editor is processed and any changed values are
  /// propagated through the graph. Finally, on deInit, the current graph layout is saved and the
  /// node editor is destroyed.
  void init() override;
  void deInit() override;
  void update() override;

 private:
  void onLoad();
  void onSave();
  void setupNodeEditor(uint16_t port);

  Settings                                     mPluginSettings;
  std::unique_ptr<csl::nodeeditor::NodeEditor> mNodeEditor;

  int mOnLoadConnection = -1;
  int mOnSaveConnection = -1;
};

} // namespace csp::demonodeeditor

#endif // CSP_DEMO_NODE_EDITOR_PLUGIN_HPP
