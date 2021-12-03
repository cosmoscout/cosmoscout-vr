////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "Plugin.hpp"
#include "TextureOverlayRenderer.hpp"

#include "../../../src/cs-core/GuiManager.hpp"
#include "../../../src/cs-core/InputManager.hpp"
#include "../../../src/cs-core/SolarSystem.hpp"
#include "../../../src/cs-core/TimeControl.hpp"

#include <VistaKernel/DisplayManager/VistaDisplayManager.h>
#include <VistaKernel/VistaSystem.h>

#include <utility>

////////////////////////////////////////////////////////////////////////////////////////////////////

EXPORT_FN cs::core::PluginBase* create() {
  return new csp::wcsoverlays::Plugin;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

EXPORT_FN void destroy(cs::core::PluginBase* pluginBase) {
  delete pluginBase; // NOLINT(cppcoreguidelines-owning-memory)
}

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace csp::wcsoverlays {

////////////////////////////////////////////////////////////////////////////////////////////////////

NLOHMANN_JSON_SERIALIZE_ENUM(WebCoverageService::CacheMode,
    {
        {WebCoverageService::CacheMode::eAlways, "always"},
        {WebCoverageService::CacheMode::eUpdateSequence, "updateSequence"},
        {WebCoverageService::CacheMode::eNever, "never"},
    })

////////////////////////////////////////////////////////////////////////////////////////////////////

void from_json(nlohmann::json const& j, Bounds& o) {
  std::array<double, 4> bounds{};
  j.get_to(bounds);
  o.mMinLon = bounds[0];
  o.mMaxLon = bounds[1];
  o.mMinLat = bounds[2];
  o.mMaxLat = bounds[3];
}

void to_json(nlohmann::json& j, Bounds const& o) {
  std::array<double, 4> bounds{o.mMinLon, o.mMaxLon, o.mMinLat, o.mMaxLat};
  j = bounds;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void from_json(nlohmann::json const& j, Plugin::Settings::Body& o) {
  cs::core::Settings::deserialize(j, "activeServer", o.mActiveServer);
  cs::core::Settings::deserialize(j, "activeBounds", o.mActiveBounds);
  cs::core::Settings::deserialize(j, "wcs", o.mWcs);
}

void to_json(nlohmann::json& j, Plugin::Settings::Body const& o) {
  cs::core::Settings::serialize(j, "activeServer", o.mActiveServer);
  cs::core::Settings::serialize(j, "activeBounds", o.mActiveBounds);
  cs::core::Settings::serialize(j, "wcs", o.mWcs);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void from_json(nlohmann::json const& j, Plugin::Settings& o) {
  cs::core::Settings::deserialize(j, "preFetch", o.mPrefetchCount);
  cs::core::Settings::deserialize(j, "maxTextureSize", o.mMaxTextureSize);
  cs::core::Settings::deserialize(j, "coverageCache", o.mCoverageCache);
  cs::core::Settings::deserialize(j, "capabilityCache", o.mCapabilityCache);
  cs::core::Settings::deserialize(j, "useCapabilityCache", o.mUseCapabilityCache);
  cs::core::Settings::deserialize(j, "bodies", o.mBodies);
  cs::core::Settings::deserialize(j, "wcsRequestFormat", o.mWcsRequestFormat);
}

void to_json(nlohmann::json& j, Plugin::Settings const& o) {
  cs::core::Settings::serialize(j, "preFetch", o.mPrefetchCount);
  cs::core::Settings::serialize(j, "maxTextureSize", o.mMaxTextureSize);
  cs::core::Settings::serialize(j, "coverageCache", o.mCoverageCache);
  cs::core::Settings::serialize(j, "capabilityCache", o.mCapabilityCache);
  cs::core::Settings::serialize(j, "useCapabilityCache", o.mUseCapabilityCache);
  cs::core::Settings::serialize(j, "bodies", o.mBodies);
  cs::core::Settings::serialize(j, "wcsRequestFormat", o.mWcsRequestFormat);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::init() {
  logger().info("Loading plugin...");

  mOnLoadConnection = mAllSettings->onLoad().connect([this]() { onLoad(); });

  mOnSaveConnection = mAllSettings->onSave().connect(
      [this]() { mAllSettings->mPlugins["csp-wcs-overlays"] = *mPluginSettings; });

  mGuiManager->addPluginTabToSideBarFromHTML(
      "WCS Overlays", "category", "../share/resources/gui/wcs_overlays_tab.html");
  mGuiManager->addSettingsSectionToSideBarFromHTML(
      "WCS Overlays", "category", "../share/resources/gui/wcs_settings.html");
  mGuiManager->addScriptToGuiFromJS("../share/resources/gui/js/csp-wcs-overlays.js");
  mGuiManager->addCssToGui("css/csp-wcs-overlays.css");

  // Fill the dropdowns with information for the active body.
  mActiveBodyConnection = mSolarSystem->pActiveBody.connectAndTouch(
      [this](std::shared_ptr<cs::scene::CelestialBody> const& body) {
        if (!body) {
          return;
        }

        auto overlay = mWCSOverlays.find(body->getCenterName());

        mGuiManager->getGui()->callJavascript(
            "CosmoScout.sidebar.setTabEnabled", "WCS Overlays", true);

        if (overlay == mWCSOverlays.end()) {
          mActiveOverlay = nullptr;
          return;
        }
        mActiveOverlay = overlay->second;

        mGuiManager->getGui()->callJavascript(
            "CosmoScout.gui.clearDropdown", "wcsOverlays.setServer");
        mGuiManager->getGui()->callJavascript(
            "CosmoScout.gui.addDropdownValue", "wcsOverlays.setServer", "None", "None", false);

        auto const& settings   = getBodySettings(overlay->second);
        bool        noneActive = true;
        for (auto const& server : mWcs[body->getCenterName()]) {
          bool active = server.getTitle() == settings.mActiveServer.get();

          mGuiManager->getGui()->callJavascript("CosmoScout.gui.addDropdownValue",
              "wcsOverlays.setServer", server.getTitle(), server.getTitle(), false);
          if (active) {
            noneActive = false;
            setWCSServer(overlay->second, server.getTitle());
          }
        }
        if (noneActive) {
          resetWCSServer(overlay->second);
        }
      });

  // Check if the observer stopped moving.
  mObserverSpeedConnection = mSolarSystem->pCurrentObserverSpeed.connect([this](float speed) {
    if (speed == 0.F) {
      mNoMovementSince           = std::chrono::high_resolution_clock::now();
      mNoMovement                = true;
      mNoMovementRequestedUpdate = false;
    } else {
      mNoMovement = false;
    }
  });

  registerSettingCallbacks();
  registerSidebarCallbacks();

  onLoad();

  logger().info("Loading done.");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::deInit() {
  logger().info("Unloading plugin...");

  mSolarSystem->pActiveBody.disconnect(mActiveBodyConnection);
  mSolarSystem->pCurrentObserverSpeed.disconnect(mObserverSpeedConnection);

  mGuiManager->removePluginTab("WCS Overlays");
  mGuiManager->removeSettingsSection("WCS Overlays");

  mGuiManager->getGui()->callJavascript(
      "CosmoScout.gui.unregisterCss", "css/csp-simple-wcs-bodies.css");

  mGuiManager->getGui()->unregisterCallback("wcsOverlays.setEnableAutomaticBoundsUpdate");
  mGuiManager->getGui()->unregisterCallback("wcsOverlays.setMaxTextureSize");
  mGuiManager->getGui()->unregisterCallback("wcsOverlays.setPrefetchCount");
  mGuiManager->getGui()->unregisterCallback("wcsOverlays.setUpdateBoundsDelay");

  mGuiManager->getGui()->unregisterCallback("wcsOverlays.setServer");
  mGuiManager->getGui()->unregisterCallback("wcsOverlays.setLayer");
  mGuiManager->getGui()->unregisterCallback("wcsOverlays.setStyle");

  mGuiManager->getGui()->unregisterCallback("wcsOverlays.goToFirstTime");
  mGuiManager->getGui()->unregisterCallback("wcsOverlays.goToPreviousTime");
  mGuiManager->getGui()->unregisterCallback("wcsOverlays.goToNextTime");
  mGuiManager->getGui()->unregisterCallback("wcsOverlays.goToLastTime");

  mGuiManager->getGui()->unregisterCallback("wcsOverlays.updateBounds");
  mGuiManager->getGui()->unregisterCallback("wcsOverlays.resetBounds");
  mGuiManager->getGui()->unregisterCallback("wcsOverlays.goToDefaultBounds");
  mGuiManager->getGui()->unregisterCallback("wcsOverlays.goToCurrentBounds");

  mAllSettings->onLoad().disconnect(mOnLoadConnection);
  mAllSettings->onSave().disconnect(mOnSaveConnection);

  logger().info("Unloading done.");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::update() {
  std::vector<std::string> finishedBodies;

  for (auto const& creationThreads : mWcsCreationThreads) {
    int running  = creationThreads.second.getRunningTaskCount();
    int total    = static_cast<int>(mPluginSettings->mBodies.at(creationThreads.first).mWcs.size());
    int progress = total - running;
    if (progress > mWcsCreationProgress.at(creationThreads.first)) {
      logger().info(
          "Loaded {} of {} WCS servers for {}...", progress, total, creationThreads.first);
      mWcsCreationProgress.at(creationThreads.first) = progress;
    }

    if (creationThreads.second.hasFinished()) {
      initOverlay(creationThreads.first, mPluginSettings->mBodies.at(creationThreads.first));
      finishedBodies.push_back(creationThreads.first);
      logger().info("Finished loading WCS servers for {}.", creationThreads.first);
    }
  }

  for (auto const& body : finishedBodies) {
    mWcsCreationThreads.erase(body);
  }

  if (mPluginSettings->mEnableAutomaticBoundsUpdate.get() && mNoMovement &&
      !mNoMovementRequestedUpdate &&
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::high_resolution_clock::now() - mNoMovementSince)
              .count() >= mPluginSettings->mUpdateBoundsDelay.get()) {
    mNoMovementRequestedUpdate = true;

    if (mActiveOverlay) {
      mActiveOverlay->requestUpdateBounds();
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::onLoad() {
  // Read settings from JSON.
  from_json(mAllSettings->mPlugins.at("csp-wcs-overlays"), *mPluginSettings);

  // First try to re-configure existing WCS overlays. We assume that they are similar if they
  // have the same name in the settings (which means they are attached to an anchor with the same
  // name).
  auto wcsOverlay = mWCSOverlays.begin();
  while (wcsOverlay != mWCSOverlays.end()) {
    auto settings = mPluginSettings->mBodies.find(wcsOverlay->first);
    if (settings != mPluginSettings->mBodies.end()) {
      // If there are settings for this overlay, reconfigure it.
      if (!settings->second.mActiveServer.isDefault()) {
        setWCSServer(wcsOverlay->second, settings->second.mActiveServer.get());
      } else {
        resetWCSServer(wcsOverlay->second);
      }

      wcsOverlay->second->configure(settings->second);

      ++wcsOverlay;
    } else {
      // Else delete it.
      wcsOverlay = mWCSOverlays.erase(wcsOverlay);
    }
  }

  // Then add new WCS overlays.
  for (auto& settings : mPluginSettings->mBodies) {
    if (mWCSOverlays.find(settings.first) != mWCSOverlays.end()) {
      continue;
    }

    auto anchor = mAllSettings->mAnchors.find(settings.first);

    if (anchor == mAllSettings->mAnchors.end()) {
      throw std::runtime_error(
          "There is no Anchor \"" + settings.first + "\" defined in the settings.");
    }

    auto wcsOverlay = std::make_shared<TextureOverlayRenderer>(
        settings.first, mSolarSystem, mTimeControl, mAllSettings, mPluginSettings, mGuiManager);

    mWCSOverlays.emplace(settings.first, wcsOverlay);

    mWcsCreationThreads.emplace(settings.first, settings.second.mWcs.size());
    mWcsCreationProgress.emplace(settings.first, 0);

    for (auto const& wcsUrl : settings.second.mWcs) {
      mWcsCreationThreads.at(settings.first).enqueue([this, settings, wcsUrl]() {
        try {
          WebCoverageService           wcs(wcsUrl, mPluginSettings->mUseCapabilityCache.get(),
              mPluginSettings->mCapabilityCache.get());
          std::unique_lock<std::mutex> lock(mWcsInsertMutex);
          mWcs[settings.first].push_back(std::move(wcs));

        } catch (std::exception const& e) {
          logger().warn("Failed to parse capabilities for '{}': '{}'!", wcsUrl, e.what());
        }
      });
    }
  }

  mSolarSystem->pActiveBody.touch(mActiveBodyConnection);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

Plugin::Settings::Body& Plugin::getBodySettings(
    std::shared_ptr<TextureOverlayRenderer> const& wcsOverlay) const {
  return mPluginSettings->mBodies.at(wcsOverlay->getCenter());
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::initOverlay(std::string const& bodyName, Settings::Body& settings) {
  auto overlay = mWCSOverlays.at(bodyName);

  if (!settings.mActiveServer.isDefault()) {
    setWCSServer(overlay, settings.mActiveServer.get());
  } else {
    resetWCSServer(overlay);
  }

  overlay->configure(settings);

  overlay->pBounds.connectAndTouch([this, &settings, center = bodyName](Bounds bounds) {
    settings.mActiveBounds = bounds;
    if (isActiveOverlay(center)) {
      mGuiManager->getGui()->callJavascript("CosmoScout.wcsOverlays.setCurrentBounds",
          bounds.mMinLon, bounds.mMaxLon, bounds.mMinLat, bounds.mMaxLat);

      if (!mActiveCoverages[mActiveOverlay->getCenter()]) {
        return;
      }
    }
  });

  if (mSolarSystem->pActiveBody.get() &&
      isActiveOverlay(mSolarSystem->pActiveBody.get()->getCenterName())) {
    mSolarSystem->pActiveBody.touch(mActiveBodyConnection);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::setWCSServer(
    std::shared_ptr<TextureOverlayRenderer> const& wcsOverlay, std::string const& name) {

  auto&       settings = getBodySettings(wcsOverlay);
  auto const& server =
      std::find_if(mWcs.at(wcsOverlay->getCenter()).begin(), mWcs.at(wcsOverlay->getCenter()).end(),
          [&name](WebCoverageService const& wcs) { return wcs.getTitle() == name; });

  if (server == mWcs.at(wcsOverlay->getCenter()).end()) {
    if (name != "None") {
      logger().warn("No server with name '{}' found!", name);
    }
    resetWCSServer(wcsOverlay);
    return;
  }

  settings.mActiveServer = name;
  mActiveServers[wcsOverlay->getCenter()].emplace(*server);

  if (isActiveOverlay(wcsOverlay)) {
    mGuiManager->getGui()->callJavascript(
        "CosmoScout.gui.setDropdownValue", "wcsOverlays.setServer", server->getTitle(), false);
    mGuiManager->getGui()->callJavascript("CosmoScout.wcsOverlays.resetLayerSelect");
  }

  bool noneActive = true;

  for (auto const& coverage : server->getCoverages()) {
    if (addCoverageToSelect(wcsOverlay, coverage, settings.mActiveCoverage.get())) {
      noneActive = false;
    }
  }
  if (isActiveOverlay(wcsOverlay)) {
    mGuiManager->getGui()->callJavascript("CosmoScout.wcsOverlays.refreshLayerSelect");
  }
  if (noneActive) {
    resetWCSCoverage(wcsOverlay);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::resetWCSServer(std::shared_ptr<TextureOverlayRenderer> const& wcsOverlay) {
  if (isActiveOverlay(wcsOverlay)) {
    mGuiManager->getGui()->callJavascript(
        "CosmoScout.gui.setDropdownValue", "wcsOverlays.setServer", "None", false);
    mGuiManager->getGui()->callJavascript("CosmoScout.wcsOverlays.resetLayerSelect");
  }

  auto& settings = getBodySettings(wcsOverlay);
  settings.mActiveServer.reset();
  mActiveServers[wcsOverlay->getCenter()].reset();
  resetWCSCoverage(wcsOverlay);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::setWCSCoverage(
    std::shared_ptr<TextureOverlayRenderer> const& wcsOverlay, std::string const& coverageId) {

  auto&                      settings = getBodySettings(wcsOverlay);
  std::optional<WebCoverage> coverage =
      mActiveServers[wcsOverlay->getCenter()]->getCoverage(coverageId);

  if (!coverage.has_value()) {
    if (coverageId != "None") {
      logger().warn("Can't set coverage '{}': No such coverage found for server '{}'", coverageId,
          mActiveServers[wcsOverlay->getCenter()]->getTitle());
    }
    resetWCSCoverage(wcsOverlay);
    return;
  }

  coverage->update();

  settings.mActiveCoverage = coverageId;
  mActiveCoverages[wcsOverlay->getCenter()].emplace(coverage.value());
  wcsOverlay->setActiveWCS(mActiveServers[wcsOverlay->getCenter()].value(),
      mActiveCoverages[wcsOverlay->getCenter()].value());

  if (isActiveOverlay(wcsOverlay)) {
    mGuiManager->getGui()->callJavascript(
        "CosmoScout.gui.setDropdownValue", "wcsOverlays.setCoverage", coverage->getId(), false);

    mGuiManager->getGui()->callJavascript("CosmoScout.wcsOverlays.setInfo", coverage->getTitle(),
        boost::replace_all_copy(
            coverage->getAbstract().value_or("<em>No description given</em>"), "\r", "</br>"),
        coverage->getSettings().mAttribution.value_or("None"),
        coverage->getKeywords().value_or("-"));
    mGuiManager->getGui()->callJavascript("CosmoScout.wcsOverlays.enableInfoButton", true);

    mGuiManager->getGui()->callJavascript("CosmoScout.wcsOverlays.enableTimeNavigation",
        !coverage->getSettings().mTimeIntervals.empty());
    mGuiManager->getGui()->callJavascript("CosmoScout.wcsOverlays.setDefaultBounds",
        coverage->getSettings().mBounds.mMinLon, coverage->getSettings().mBounds.mMaxLon,
        coverage->getSettings().mBounds.mMinLat, coverage->getSettings().mBounds.mMaxLat);
    mGuiManager->getGui()->callJavascript("CosmoScout.wcsOverlays.setCurrentBounds",
        wcsOverlay->pBounds.get().mMinLon, wcsOverlay->pBounds.get().mMaxLon,
        wcsOverlay->pBounds.get().mMinLat, wcsOverlay->pBounds.get().mMaxLat);

    if (!coverage->getSettings().mTimeIntervals.empty()) {
      auto intervals = coverage->getSettings().mTimeIntervals;
      auto start = utils::timeToString(intervals.begin()->mFormat, intervals.begin()->mStartTime);
      auto end   = utils::timeToString(intervals.begin()->mFormat, intervals.begin()->mEndTime);

      mGuiManager->getGui()->callJavascript("CosmoScout.wcsOverlays.setTimeInfo", start, end);
    } else {
      mGuiManager->getGui()->callJavascript("CosmoScout.wcsOverlays.setTimeInfo", "", "");
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::resetWCSCoverage(std::shared_ptr<TextureOverlayRenderer> const& wcsOverlay) {
  if (isActiveOverlay(wcsOverlay)) {
    mGuiManager->getGui()->callJavascript(
        "CosmoScout.gui.setDropdownValue", "wcsOverlays.setCoverage", "None", false);

    mGuiManager->getGui()->callJavascript("CosmoScout.wcsOverlays.enableInfoButton", false);
    mGuiManager->getGui()->callJavascript("CosmoScout.wcsOverlays.clearDefaultBounds");
    mGuiManager->getGui()->callJavascript("CosmoScout.wcsOverlays.clearCurrentBounds");
    mGuiManager->getGui()->callJavascript("CosmoScout.wcsOverlays.enableTimeNavigation", false);

    mGuiManager->getGui()->callJavascript("CosmoScout.wcsOverlays.resetTransferFunction");
  }

  auto& settings = getBodySettings(wcsOverlay);
  settings.mActiveCoverage.reset();
  mActiveCoverages[wcsOverlay->getCenter()].reset();
  wcsOverlay->clearActiveWCS();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool Plugin::isActiveOverlay(std::shared_ptr<TextureOverlayRenderer> const& wcsOverlay) {
  return mActiveOverlay && wcsOverlay && wcsOverlay->getCenter() == mActiveOverlay->getCenter();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool Plugin::isActiveOverlay(std::string const& center) {
  return mActiveOverlay && center == mActiveOverlay->getCenter();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool Plugin::addCoverageToSelect(std::shared_ptr<TextureOverlayRenderer> const& wcsOverlay,
    const WebCoverage& coverage, std::string const& activeLayer) {
  bool active = coverage.getId() == activeLayer;

  if (isActiveOverlay(wcsOverlay)) {
    mGuiManager->getGui()->callJavascript("CosmoScout.wcsOverlays.addCoverage", coverage.getId(),
        coverage.getTitle(), active, coverage.isRequestable());
  }

  bool anyActive = active;
  if (active) {
    setWCSCoverage(wcsOverlay, coverage.getId());
  }

  return anyActive;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::goToBounds(Bounds const& bounds) {
  double lon      = (bounds.mMinLon + bounds.mMaxLon) / 2.;
  double lat      = (bounds.mMinLat + bounds.mMaxLat) / 2.;
  double lonRange = bounds.mMaxLon - bounds.mMinLon;
  double latRange = bounds.mMaxLat - bounds.mMinLat;

  VistaTransformMatrix proj =
      GetVistaSystem()->GetDisplayManager()->GetCurrentRenderInfo()->m_matProjection;
  double fovy = 2.0 * atan(1.0 / proj[1][1]);
  double fovx = 2.0 * atan(1.0 / proj[0][0]);

  // Rough approximation of the height, at which the whole bounds are in frame
  double radius = mSolarSystem->getRadii(mActiveOverlay->getCenter())[0];
  double heighty =
      std::tan(cs::utils::convert::toRadians(latRange) / 2.) * radius / std::tan(fovy / 2.);
  double heightx =
      std::tan(cs::utils::convert::toRadians(lonRange) / 2.) * radius / std::tan(fovx / 2.);
  heightx -= radius * (1 - std::cos(cs::utils::convert::toRadians(lonRange) / 2.));
  heighty -= radius * (1 - std::cos(cs::utils::convert::toRadians(latRange) / 2.));

  mSolarSystem->flyObserverTo(mSolarSystem->pActiveBody.get()->getCenterName(),
      mSolarSystem->pActiveBody.get()->getFrameName(),
      cs::utils::convert::toRadians(glm::dvec2(lon, lat)), std::max(heighty, heightx), 5.);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::registerSettingCallbacks() {

  mGuiManager->getGui()->registerCallback("wcsOverlays.setEnableAutomaticBoundsUpdate",
      "Enables or disables automatically updating the bounds when the observer stops moving.",
      std::function(
          [this](bool enable) { mPluginSettings->mEnableAutomaticBoundsUpdate = enable; }));

  mGuiManager->getGui()->registerCallback("wcsOverlays.setMaxTextureSize",
      "Set the maximum texture size for wcs requests.", std::function([this](double value) {
        mPluginSettings->mMaxTextureSize = std::lround(value);
      }));

  mGuiManager->getGui()->registerCallback("wcsOverlays.setPrefetchCount",
      "Set the amount of images to prefetch in both directions of time.",
      std::function(
          [this](double value) { mPluginSettings->mPrefetchCount = std::lround(value); }));

  mGuiManager->getGui()->registerCallback("wcsOverlays.setUpdateBoundsDelay",
      "Set the delay that has to pass before the bounds are automatically updated.",
      std::function(
          [this](double value) { mPluginSettings->mUpdateBoundsDelay = std::lround(value); }));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::registerSidebarCallbacks() {
  mGuiManager->getGui()->registerCallback(
      "wcsOverlays.updateBounds", "Updates the bounds for wcs requests.", std::function([this]() {
        if (!mActiveOverlay) {
          return;
        }

        mActiveOverlay->requestUpdateBounds();
      }));

  mGuiManager->getGui()->registerCallback("wcsOverlays.resetBounds",
      "Resets the bounds for wcs requests to the current layer's default bounds.",
      std::function([this]() {
        if (!mActiveOverlay || !mActiveCoverages[mActiveOverlay->getCenter()]) {
          return;
        }

        mActiveOverlay->pBounds =
            mActiveCoverages[mActiveOverlay->getCenter()]->getSettings().mBounds;
      }));

  mGuiManager->getGui()->registerCallback("wcsOverlays.goToDefaultBounds",
      "Fly the observer to a position from which most of the current layer's default bounds is "
      "visible.",
      std::function([this]() {
        if (!mActiveOverlay || !mActiveCoverages[mActiveOverlay->getCenter()]) {
          return;
        }

        WebCoverage::Settings layerSettings =
            mActiveCoverages[mActiveOverlay->getCenter()]->getSettings();
        goToBounds(layerSettings.mBounds);
      }));

  mGuiManager->getGui()->registerCallback("wcsOverlays.goToCurrentBounds",
      "Fly the observer to a position from which most of the currently active bounds is visible.",
      std::function([this]() {
        if (!mActiveOverlay) {
          return;
        }

        goToBounds(mActiveOverlay->pBounds.get());
      }));

  mGuiManager->getGui()->registerCallback("wcsOverlays.setServer",
      "Set the current planet's WCS server to the one with the given name.",
      std::function([this](std::string&& name) {
        if (!mActiveOverlay) {
          return;
        }

        setWCSServer(mActiveOverlay, name);
        mNoMovementRequestedUpdate = false;
      }));

  mGuiManager->getGui()->registerCallback("wcsOverlays.setCoverage",
      "Set the current planet's WCS coverage to the one with the given name.",
      std::function([this](std::string&& coverageId) {
        if (!mActiveOverlay || !mActiveServers[mActiveOverlay->getCenter()]) {
          return;
        }

        setWCSCoverage(mActiveOverlay, coverageId);
        mNoMovementRequestedUpdate = false;
      }));

  mGuiManager->getGui()->registerCallback("wcsOverlays.setLayer",
      "Set the current layer of the active Coverage.", std::function([this](std::string&& layer) {
        if (!mActiveOverlay || !mActiveServers[mActiveOverlay->getCenter()]) {
          return;
        }

        mActiveOverlay->setLayer(std::stoi(layer));
      }));

  mGuiManager->getGui()->registerCallback(
      "wcsOverlays.goToFirstTime", "Go to the first available timestep.", std::function([this]() {
        if (!mActiveOverlay || !mActiveCoverages[mActiveOverlay->getCenter()] ||
            mActiveCoverages[mActiveOverlay->getCenter()]->getSettings().mTimeIntervals.empty()) {
          return;
        }

        mActiveCoverages[mActiveOverlay->getCenter()]->update();
        auto intervals =
            mActiveCoverages[mActiveOverlay->getCenter()]->getSettings().mTimeIntervals;
        auto start = utils::timeToString(intervals.front().mFormat, intervals.front().mStartTime);
        auto end   = utils::timeToString(intervals.back().mFormat, intervals.back().mEndTime);

        mGuiManager->getGui()->callJavascript("CosmoScout.wcsOverlays.setTimeInfo", start, end);

        mAllSettings->pTimeSpeed = 0.f;
        mTimeControl->setTime(
            cs::utils::convert::time::toSpice(mActiveCoverages[mActiveOverlay->getCenter()]
                                                  ->getSettings()
                                                  .mTimeIntervals.front()
                                                  .mStartTime));
      }));

  mGuiManager->getGui()->registerCallback("wcsOverlays.goToPreviousTime",
      "Go to the previous available timestep.", std::function([this]() {
        if (!mActiveOverlay || !mActiveCoverages[mActiveOverlay->getCenter()] ||
            mActiveCoverages[mActiveOverlay->getCenter()]->getSettings().mTimeIntervals.empty()) {
          return;
        }

        mAllSettings->pTimeSpeed = 0.f;

        boost::posix_time::ptime time =
            cs::utils::convert::time::toPosix(mTimeControl->pSimulationTime.get());

        std::vector<TimeInterval> intervals =
            mActiveCoverages[mActiveOverlay->getCenter()]->getSettings().mTimeIntervals;

        // Check if current time is in any interval
        TimeInterval             result;
        boost::posix_time::ptime sampleStartTime = time;
        if (utils::timeInIntervals(sampleStartTime, intervals, result)) {
          if (sampleStartTime != time) {
            // timeInIntervals rounds down the time to the nearest timestep, so the
            // result of that method can be used.
            mTimeControl->setTime(cs::utils::convert::time::toSpice(sampleStartTime));
            return;
          }
          // The current time was a valid timestep so the previous step has to be found.
          if (sampleStartTime == result.mStartTime) {
            auto it = std::find(intervals.begin(), intervals.end(), result);
            if (it == intervals.begin()) {
              // If the time is at the start of the first interval, there is no previous
              // timestep to go to.
              return;
            }
            // If the time is at the start of another interval, the previous timestep is the
            // end time of the previous interval.
            // It is assumed that the intervals are ordered chronologically.
            mTimeControl->setTime(cs::utils::convert::time::toSpice((it - 1)->mEndTime));
            return;
          }
          // If the time was not the start time of any interval we can subtract the duration to
          // get the previous timestep.
          sampleStartTime = utils::addDurationToTime(sampleStartTime, result.mSampleDuration, -1);
          mTimeControl->setTime(cs::utils::convert::time::toSpice(sampleStartTime));
          return;
        }

        // Time was not part of any interval, so the last interval, that lies before the current
        // time has to be found.
        boost::posix_time::ptime temp = time;
        for (auto const& interval : intervals) {
          if (time > interval.mEndTime) {
            temp = interval.mEndTime;
          } else if (time < interval.mStartTime) {
            break;
          }
        }
        mTimeControl->setTime(cs::utils::convert::time::toSpice(temp));
      }));

  mGuiManager->getGui()->registerCallback(
      "wcsOverlays.goToNextTime", "Go to the next available timestep.", std::function([this]() {
        if (!mActiveOverlay || !mActiveCoverages[mActiveOverlay->getCenter()] ||
            mActiveCoverages[mActiveOverlay->getCenter()]->getSettings().mTimeIntervals.empty()) {
          return;
        }

        mAllSettings->pTimeSpeed = 0.f;

        boost::posix_time::ptime time =
            cs::utils::convert::time::toPosix(mTimeControl->pSimulationTime.get());

        std::vector<TimeInterval> intervals =
            mActiveCoverages[mActiveOverlay->getCenter()]->getSettings().mTimeIntervals;

        // Check if current time is in any interval
        TimeInterval             result;
        boost::posix_time::ptime sampleStartTime = time;
        if (utils::timeInIntervals(sampleStartTime, intervals, result)) {
          if (sampleStartTime == result.mEndTime) {
            auto it = std::find(intervals.begin(), intervals.end(), result);
            if (it == intervals.end() - 1) {
              // If the time is at the end of the last interval, there is no next
              // timestep to go to.
              return;
            }
            // If the time is at the end of another interval, the next timestep is the
            // start time of the next interval.
            // It is assumed that the intervals are ordered chronologically.
            mTimeControl->setTime(cs::utils::convert::time::toSpice((it + 1)->mStartTime));
            return;
          }
          // If the time was not the end time of any interval we can add the duration to
          // get the next timestep.
          sampleStartTime = utils::addDurationToTime(sampleStartTime, result.mSampleDuration);
          mTimeControl->setTime(cs::utils::convert::time::toSpice(sampleStartTime));
          return;
        }

        // Time was not part of any interval, so the first interval, that lies after the current
        // time has to be found.
        for (auto const& interval : intervals) {
          if (time < interval.mStartTime) {
            mTimeControl->setTime(cs::utils::convert::time::toSpice(interval.mStartTime));
            return;
          }
        }
      }));

  mGuiManager->getGui()->registerCallback(
      "wcsOverlays.goToLastTime", "Go to the last available timestep.", std::function([this]() {
        if (!mActiveOverlay || !mActiveCoverages[mActiveOverlay->getCenter()] ||
            mActiveCoverages[mActiveOverlay->getCenter()]->getSettings().mTimeIntervals.empty()) {
          return;
        }

        mActiveCoverages[mActiveOverlay->getCenter()]->update();
        auto intervals =
            mActiveCoverages[mActiveOverlay->getCenter()]->getSettings().mTimeIntervals;
        auto start = utils::timeToString(intervals.front().mFormat, intervals.front().mStartTime);
        auto end   = utils::timeToString(intervals.back().mFormat, intervals.back().mEndTime);

        mGuiManager->getGui()->callJavascript("CosmoScout.wcsOverlays.setTimeInfo", start, end);

        mAllSettings->pTimeSpeed = 0.f;
        mTimeControl->setTime(
            cs::utils::convert::time::toSpice(mActiveCoverages[mActiveOverlay->getCenter()]
                                                  ->getSettings()
                                                  .mTimeIntervals.back()
                                                  .mEndTime));
      }));

  // Callback to set a transfer function for the rendering
  mGuiManager->getGui()->registerCallback("wcsOverlays.setTransferFunction",
      "Sets the transfer function for rendering", std::function([this](std::string val) {
        if (!mActiveOverlay) {
          return;
        }

        mActiveOverlay->setTransferFunction(val);
      }));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::wcsoverlays
