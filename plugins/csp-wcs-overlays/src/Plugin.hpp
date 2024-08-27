////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSP_WCS_OVERLAYS_PLUGIN_HPP
#define CSP_WCS_OVERLAYS_PLUGIN_HPP

#include "../../csl-ogc/src/wcs/WebCoverageService.hpp"
#include "../../csl-ogc/src/common/utils.hpp"

#include "../../../src/cs-core/PluginBase.hpp"
#include "../../../src/cs-utils/DefaultProperty.hpp"
#include "../../../src/cs-utils/ThreadPool.hpp"

#include <chrono>
#include <map>
#include <string>

namespace csp::wcsoverlays {

class TextureOverlayRenderer;

/// This plugin provides the rendering of planets as spheres with a texture and an additional WCS
/// based texture. Despite its name it can also render moons :P. It can be configured via the
/// applications config file. See README.md for details.
class Plugin : public cs::core::PluginBase {
 public:
  /// The startup settings of the plugin.
  struct Settings {
    /// Specifies whether to automatically update the overlay bounds when the observer stopped
    /// moving for a certain amount of time.
    cs::utils::DefaultProperty<bool> mEnableAutomaticBoundsUpdate{false};

    /// Path to the coverage cache folder, can be absolute or relative to the cosmoscout executable.
    cs::utils::DefaultProperty<std::string> mCoverageCache{
        "../share/cache/csp-wcs-overlays/texture-cache"};

    /// Path to the wcs capability cache folder, can be absolute or relative to the cosmoscout
    /// executable.
    cs::utils::DefaultProperty<std::string> mCapabilityCache{
        "../share/cache/csp-wcs-overlays/wcs-capability-cache"};

    /// Specifies whether cached capabilitiy files should be used instead of requesting new ones.
    cs::utils::DefaultProperty<csl::ogc::WebCoverageService::CacheMode> mUseCapabilityCache{
        csl::ogc::WebCoverageService::CacheMode::eNever};

    /// The amount of textures that gets pre-fetched in every time direction.
    cs::utils::DefaultProperty<int> mPrefetchCount{0};

    /// The size of the requested coverage textures along the longer axis.
    cs::utils::DefaultProperty<int> mMaxTextureSize{1024};

    /// If automatic bounds update is enabled, the bounds will be updated when the observer stopped
    /// moving for this amount of milliseconds.
    cs::utils::DefaultProperty<int> mUpdateBoundsDelay{1000};

    /// The format used in wcs calls, defaults to .tiff files
    /// Available formats are usually found on the WCS GetCapabilities call
    cs::utils::DefaultProperty<std::string> mWcsRequestFormat{"image/tiff"};

    /// The startup settings for a planet.
    struct Body {
      /// The name of the currently active WCS server.
      cs::utils::DefaultProperty<std::string> mActiveServer{"None"};
      /// The name of the currently active WCS coverage.
      cs::utils::DefaultProperty<std::string> mActiveCoverage{"None"};
      /// The bounds for the currently active WCS layer.
      cs::utils::DefaultProperty<csl::ogc::Bounds2D> mActiveBounds{{-180., 180., -90., 90.}};
      ///	URLs of WCS servers.
      std::vector<std::string> mWcs;
    };

    /// A list of bodies with their anchor names.
    std::map<std::string, Body> mBodies;
  };

  void init() override;
  void deInit() override;

  void update() override;

 private:
  void onLoad();

  Settings::Body& getBodySettings(std::shared_ptr<TextureOverlayRenderer> const& wcsOverlay) const;

  void initOverlay(std::string const& bodyName, Settings::Body& settings);

  void setWCSServer(
      std::shared_ptr<TextureOverlayRenderer> const& wcsOverlay, std::string const& name);
  void resetWCSServer(std::shared_ptr<TextureOverlayRenderer> const& wcsOverlay);
  void setWCSCoverage(
      std::shared_ptr<TextureOverlayRenderer> const& wcsOverlay, std::string const& coverageId);
  void resetWCSCoverage(std::shared_ptr<TextureOverlayRenderer> const& wcsOverlay);

  /// Checks if the given wcsOverlay is currently active.
  bool isActiveOverlay(std::shared_ptr<TextureOverlayRenderer> const& wcsOverlay);
  /// Checks if the wcsOverlay with the given center is currently active.
  bool isActiveOverlay(std::string const& center);

  /// Adds the given coverage and all of its sublayers to the coverage dropdown.
  /// Returns whether any coverage's name matched the given activeLayer.
  bool addCoverageToSelect(std::shared_ptr<TextureOverlayRenderer> const& wcsOverlay,
      const csl::ogc::WebCoverage& coverage, std::string const& activeLayer);

  /// Move the observer so that the given bounds are visible.
  void goToBounds(csl::ogc::Bounds2D const& bounds);

  /// Gui Callbacks used in WCS settings
  void registerSettingCallbacks();
  /// Gui Callbacks used in WCS sidebar
  void registerSidebarCallbacks();

  std::shared_ptr<Settings> mPluginSettings = std::make_shared<Settings>();

  std::mutex                                   mWcsInsertMutex;
  std::map<std::string, cs::utils::ThreadPool> mWcsCreationThreads;
  std::map<std::string, int>                   mWcsCreationProgress;

  std::map<std::string, std::shared_ptr<TextureOverlayRenderer>> mWCSOverlays;
  std::map<std::string, std::vector<csl::ogc::WebCoverageService>>         mWcs;

  std::shared_ptr<TextureOverlayRenderer> mActiveOverlay;
  /// The currently active WebCoverageService for each center name.
  std::map<std::string, std::optional<csl::ogc::WebCoverageService>> mActiveServers;
  /// The currently active WebMapLayer for each center name.
  std::map<std::string, std::optional<csl::ogc::WebCoverage>> mActiveCoverages;

  /// True when the observer is not moving.
  bool mNoMovement{};
  /// True when the active overlay was requested to update its bounds
  /// because the observer is not moving.
  bool mNoMovementRequestedUpdate{};
  /// Time at which the observer stopped moving.
  std::chrono::time_point<std::chrono::high_resolution_clock> mNoMovementSince;

  int mActiveBodyConnection    = -1;
  int mObserverSpeedConnection = -1;
  int mOnLoadConnection        = -1;
  int mOnSaveConnection        = -1;
};

} // namespace csp::wcsoverlays

#endif // CSP_WCS_OVERLAYS_PLUGIN_HPP