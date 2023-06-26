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

 private:
  void onLoad();
  void onSave();

  std::shared_ptr<Settings> mPluginSettings = std::make_shared<Settings>();
  std::unique_ptr<FeatureRenderer> mRenderer;

  // Below, some lines of the WMS file that could be useful.
  /*
  void setWMSServer(
      std::shared_ptr<TextureOverlayRenderer> const& wmsOverlay, std::string const& name);
  void resetWMSServer(std::shared_ptr<TextureOverlayRenderer> const& wmsOverlay);
  void setWMSLayer(
      std::shared_ptr<TextureOverlayRenderer> const& wmsOverlay, std::string const& name);
  void resetWMSLayer(std::shared_ptr<TextureOverlayRenderer> const& wmsOverlay);
  void setWMSStyle(
      std::shared_ptr<TextureOverlayRenderer> const& wmsOverlay, std::string const& name);
  void resetWMSStyle(std::shared_ptr<TextureOverlayRenderer> const& wmsOverlay);

  /// Checks if the given wmsOverlay is currently active.
  bool isActiveOverlay(std::shared_ptr<TextureOverlayRenderer> const& wmsOverlay);
  /// Checks if the wmsOverlay with the given center is currently active.
  bool isActiveOverlay(std::string const& center);
  */

 /* 
 std::shared_ptr<TextureOverlayRenderer> mActiveOverlay;
  /// The currently active WebMapService for each center name.
  std::map<std::string, std::optional<WebMapService>> mActiveServers;
  /// The currently active WebMapLayer for each center name.
  std::map<std::string, std::optional<WebMapLayer>> mActiveLayers;

  /// True when the observer is not moving.
  bool mNoMovement{};
  /// True when the active overlay was requested to update its bounds
  /// because the observer is not moving.
  bool mNoMovementRequestedUpdate{};
  /// Time at which the observer stopped moving.
  std::chrono::time_point<std::chrono::high_resolution_clock> mNoMovementSince;
 */

  int mOnLoadConnection       = -1;
  int mOnSaveConnection       = -1;
};



} // namespace csp::wfsoverlays

#endif // CSP_WFS_OVERLAYS_PLUGIN_HPP
