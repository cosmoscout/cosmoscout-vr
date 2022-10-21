////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSP_WEB_API_PLUGIN_HPP
#define CSP_WEB_API_PLUGIN_HPP

#include "../../../src/cs-core/PluginBase.hpp"
#include "../../../src/cs-utils/Property.hpp"
#include "../../csl-node-editor/src/NodeEditor.hpp"

namespace csp::demonodeeditor {

/// This plugin contains a web server which provides some HTTP endpoints which can be used to
/// remote-control CosmoScout VR.
class Plugin : public cs::core::PluginBase {
 public:
  struct Settings {
    /// The port where the server should listen on. For example 9999.
    cs::utils::Property<uint16_t> mPort;
  };

  void init() override;
  void deInit() override;

  void update() override;

 private:
  void onLoad();
  void onSave();

  Settings                                     mPluginSettings;
  std::unique_ptr<csl::nodeeditor::NodeEditor> mNodeEditor;

  int mOnLoadConnection = -1;
  int mOnSaveConnection = -1;
};

} // namespace csp::demonodeeditor

#endif // CSP_WEB_API_PLUGIN_HPP
