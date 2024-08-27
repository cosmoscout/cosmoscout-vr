////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSP_VISUAL_QUERY_PLUGIN_HPP
#define CSP_VISUAL_QUERY_PLUGIN_HPP

#include "../../../src/cs-core/PluginBase.hpp"
#include "../../../src/cs-core/Settings.hpp"
#include "../../../src/cs-utils/Property.hpp"
#include "../../csl-node-editor/src/NodeEditor.hpp"
#include "../../csl-ogc/src/wcs/WebCoverageService.hpp"

#include <memory>

namespace csp::visualquery {

/// Your plugin description here!
class Plugin : public cs::core::PluginBase {
 public:
  struct Settings {
    /// The port where the server should listen on. For example 9999.
    cs::utils::Property<uint16_t> mPort;

    /// Here we will store the current node graph if the scene gets saved. You could also save node
    /// graphs to separate json files instead and load them at runtime. The JSON format of this
    /// graph is defined in csl-node-editor/src/NodeEditor.hpp.
    std::optional<nlohmann::json> mGraph;

    // URL which contains the WCS server
    std::vector<std::string> mWcsUrl;
    std::vector<csl::ogc::WebCoverageService> mWebCoverages;
  };

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

} // namespace csp::visualquery

#endif // CSP_VISUAL_QUERY_PLUGIN_HPP