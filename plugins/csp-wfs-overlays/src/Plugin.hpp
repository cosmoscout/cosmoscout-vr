////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSP_WFS_OVERLAYS_PLUGIN_HPP
#define CSP_WFS_OVERLAYS_PLUGIN_HPP

#include "../../../src/cs-core/PluginBase.hpp"
#include "../../../src/cs-core/Settings.hpp"
#include "../../../src/cs-utils/Property.hpp"
#include <memory>
#include <unordered_set>
#include "FeatureRenderer.hpp"
#include "WFSTypes.hpp"

namespace csp::wfsoverlays {

/// This plugin represents Web Feature Servivces data  in space. The plugin is configurable via the application
/// config file. See README.md for details.

class Plugin : public cs::core::PluginBase {
  public:
  struct Settings {
    cs::utils::DefaultProperty<bool> mEnabled{true};
    std::vector<std::string> mWfs; 
  };
  
  void init() override;
  void deInit() override;
  void update() override;

  void setWFSServer(std::string URL);
  void setWFSFeatureType(std::string featureType);

 private:
  void onLoad();
  void onSave();

  std::shared_ptr<Settings> mPluginSettings = std::make_shared<Settings>();
  std::vector<std::unique_ptr<FeatureRenderer>> mRenderers;
  std::string mBaseUrl;

  
  int mOnLoadConnection       = -1;
  int mOnSaveConnection       = -1;
};

} // namespace csp::wfsoverlays

#endif // CSP_WFS_OVERLAYS_PLUGIN_HPP
