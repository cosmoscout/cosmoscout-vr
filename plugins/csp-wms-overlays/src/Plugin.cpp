////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "Plugin.hpp"
#include "TextureOverlayRenderer.hpp"
#include "WebMapService.hpp"
#include "logger.hpp"

#include "../../../src/cs-core/GuiManager.hpp"
#include "../../../src/cs-core/InputManager.hpp"
#include "../../../src/cs-core/Settings.hpp"
#include "../../../src/cs-core/SolarSystem.hpp"
#include "../../../src/cs-core/TimeControl.hpp"

#include <VistaKernel/DisplayManager/VistaDisplayManager.h>
#include <VistaKernel/VistaSystem.h>

////////////////////////////////////////////////////////////////////////////////////////////////////

EXPORT_FN cs::core::PluginBase* create() {
  return new csp::wmsoverlays::Plugin;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

EXPORT_FN void destroy(cs::core::PluginBase* pluginBase) {
  delete pluginBase; // NOLINT(cppcoreguidelines-owning-memory)
}

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace csp::wmsoverlays {

////////////////////////////////////////////////////////////////////////////////////////////////////

void from_json(nlohmann::json const& j, Plugin::Settings::Body& o) {
  cs::core::Settings::deserialize(j, "activeServer", o.mActiveServer);
  cs::core::Settings::deserialize(j, "activeLayer", o.mActiveLayer);
  cs::core::Settings::deserialize(j, "activeStyle", o.mActiveStyle);
  cs::core::Settings::deserialize(j, "wms", o.mWms);
}

void to_json(nlohmann::json& j, Plugin::Settings::Body const& o) {
  cs::core::Settings::serialize(j, "activeServer", o.mActiveServer);
  cs::core::Settings::serialize(j, "activeLayer", o.mActiveLayer);
  cs::core::Settings::serialize(j, "activeStyle", o.mActiveStyle);
  cs::core::Settings::serialize(j, "wms", o.mWms);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void from_json(nlohmann::json const& j, Plugin::Settings& o) {
  cs::core::Settings::deserialize(j, "preFetch", o.mPrefetchCount);
  cs::core::Settings::deserialize(j, "maxTextureSize", o.mMaxTextureSize);
  cs::core::Settings::deserialize(j, "mapCache", o.mMapCache);
  cs::core::Settings::deserialize(j, "capabilityCache", o.mCapabilityCache);
  cs::core::Settings::deserialize(j, "bodies", o.mBodies);
}

void to_json(nlohmann::json& j, Plugin::Settings const& o) {
  cs::core::Settings::serialize(j, "preFetch", o.mPrefetchCount);
  cs::core::Settings::serialize(j, "maxTextureSize", o.mMaxTextureSize);
  cs::core::Settings::serialize(j, "mapCache", o.mMapCache);
  cs::core::Settings::serialize(j, "capabilityCache", o.mCapabilityCache);
  cs::core::Settings::serialize(j, "bodies", o.mBodies);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::init() {
  logger().info("Loading plugin...");

  mOnLoadConnection = mAllSettings->onLoad().connect([this]() { onLoad(); });

  mOnSaveConnection = mAllSettings->onSave().connect(
      [this]() { mAllSettings->mPlugins["csp-wms-overlays"] = *mPluginSettings; });

  mGuiManager->addPluginTabToSideBarFromHTML(
      "WMS", "panorama", "../share/resources/gui/wms_overlays_tab.html");
  mGuiManager->addSettingsSectionToSideBarFromHTML(
      "WMS", "panorama", "../share/resources/gui/wms_settings.html");
  mGuiManager->addScriptToGuiFromJS("../share/resources/gui/js/csp-wms-overlays.js");
  mGuiManager->addCssToGui("css/csp-wms-overlays.css");

  // Updates the bounds for which map data is requested.
  mGuiManager->getGui()->registerCallback(
      "wmsOverlays.updateBounds", "Updates the bounds for map requests.", std::function([this]() {
        if (mActiveOverlay) {
          mActiveOverlay->requestUpdateBounds();
        }
      }));

  // Moves the observer to a position from which most of the current layer should be visible.
  mGuiManager->getGui()->registerCallback("wmsOverlays.goToDefaultBounds",
      "Fly the observer to the center of the default bounds of the current layer.",
      std::function([this]() {
        if (!mActiveOverlay || !mActiveLayers[mActiveOverlay->getCenter()]) {
          return;
        }
        WebMapLayer::Settings layerSettings =
            mActiveLayers[mActiveOverlay->getCenter()]->getSettings();

        goToBounds(layerSettings.mBounds);
      }));

  // Moves the observer to a position from which most of the currently set bounds should be visible.
  mGuiManager->getGui()->registerCallback("wmsOverlays.goToCurrentBounds",
      "Fly the observer to the center of the current bounds.", std::function([this]() {
        if (!mActiveOverlay) {
          return;
        }

        goToBounds(mActiveOverlay->pBounds.get());
      }));

  // Set whether to interpolate textures between timesteps (does not work when pre-fetch is
  // inactive).
  mGuiManager->getGui()->registerCallback("wmsOverlays.setEnableTimeInterpolation",
      "Enables or disables interpolation.",
      std::function([this](bool enable) { mPluginSettings->mEnableInterpolation = enable; }));

  // Set whether to display timespan.
  mGuiManager->getGui()->registerCallback("wmsOverlays.setEnableTimeSpan",
      "Enables or disables timespan.",
      std::function([this](bool enable) { mPluginSettings->mEnableTimespan = enable; }));

  // Set whether to automatically update bounds.
  mGuiManager->getGui()->registerCallback("wmsOverlays.setEnableAutomaticBoundsUpdate",
      "Enables or disables automatic bounds update.", std::function([this](bool enable) {
        mPluginSettings->mEnableAutomaticBoundsUpdate = enable;
      }));

  // Set WMS source.
  mGuiManager->getGui()->registerCallback("wmsOverlays.setServer",
      "Set the current planet's WMS server to the one with the given name.",
      std::function([this](std::string&& name) {
        if (mActiveOverlay) {
          setWMSServer(mActiveOverlay, name);
          mNoMovementRequestedUpdate = false;
        }
      }));

  mGuiManager->getGui()->registerCallback("wmsOverlays.setLayer",
      "Set the current planet's WMS layer to the one with the given name.",
      std::function([this](std::string&& name) {
        if (mActiveOverlay) {
          setWMSLayer(mActiveOverlay, name);
          mNoMovementRequestedUpdate = false;
        }
      }));

  mGuiManager->getGui()->registerCallback("wmsOverlays.setStyle",
      "Sets the style for the currently selected layer.", std::function([this](std::string&& name) {
        if (mActiveOverlay || mActiveLayers[mActiveOverlay->getCenter()]) {
          setWMSStyle(mActiveOverlay, name);
          mNoMovementRequestedUpdate = false;
        }
      }));

  mGuiManager->getGui()->registerCallback(
      "wmsOverlays.goToFirstTime", "Go to the first available timestep.", std::function([this]() {
        if (!mActiveOverlay || !mActiveLayers[mActiveOverlay->getCenter()] ||
            mActiveLayers[mActiveOverlay->getCenter()]->getSettings().mTimeIntervals.empty()) {
          return;
        }
        mTimeControl->setTimeSpeed(0);
        mTimeControl->setTime(
            cs::utils::convert::time::toSpice(mActiveLayers[mActiveOverlay->getCenter()]
                                                  ->getSettings()
                                                  .mTimeIntervals.front()
                                                  .mStartTime));
      }));

  mGuiManager->getGui()->registerCallback("wmsOverlays.goToPreviousTime",
      "Go to the previous available timestep.", std::function([this]() {
        if (!mActiveOverlay || !mActiveLayers[mActiveOverlay->getCenter()] ||
            mActiveLayers[mActiveOverlay->getCenter()]->getSettings().mTimeIntervals.empty()) {
          return;
        }

        mTimeControl->setTimeSpeed(0);

        boost::posix_time::ptime time =
            cs::utils::convert::time::toPosix(mTimeControl->pSimulationTime.get());

        std::vector<TimeInterval> intervals =
            mActiveLayers[mActiveOverlay->getCenter()]->getSettings().mTimeIntervals;

        // Check if current time is in any interval
        TimeInterval             result;
        boost::posix_time::ptime sampleStartTime = time;
        if (utils::timeInIntervals(sampleStartTime, intervals, result)) {
          if (sampleStartTime != time) {
            // timeInIntervals rounds down the time to the nearest timestep, so the
            // result of that method can be used.
            mTimeControl->setTime(cs::utils::convert::time::toSpice(sampleStartTime));
            return;
          } else {
            // The current time was a valid timestep so the previous step has to be found.
            if (sampleStartTime == result.mStartTime) {
              auto it = std::find(intervals.begin(), intervals.end(), result);
              if (it == intervals.begin()) {
                // If the time is at the start of the first interval, there is no previous
                // timestep to go to.
                return;
              } else {
                // If the time is at the start of another interval, the previous timestep is the
                // end time of the previous interval.
                // Currently we trust that the intervals are ordered chronologically
                mTimeControl->setTime(cs::utils::convert::time::toSpice((it - 1)->mEndTime));
                return;
              }
            }
            // If the time was not the start time of any interval we can substract the duration to
            // get the previous timestep.
            sampleStartTime = utils::addDurationToTime(sampleStartTime, result.mSampleDuration, -1);
            mTimeControl->setTime(cs::utils::convert::time::toSpice(sampleStartTime));
            return;
          }
        }

        boost::posix_time::ptime temp = time;
        for (auto const& interval : intervals) {
          if (time > interval.mEndTime) {
            temp = interval.mEndTime;
          } else if (time < interval.mStartTime) {
            mTimeControl->setTime(cs::utils::convert::time::toSpice(temp));
            return;
          }
        }
        mTimeControl->setTime(cs::utils::convert::time::toSpice(temp));
      }));

  mGuiManager->getGui()->registerCallback(
      "wmsOverlays.goToNextTime", "Go to the next available timestep.", std::function([this]() {
        if (!mActiveOverlay || !mActiveLayers[mActiveOverlay->getCenter()] ||
            mActiveLayers[mActiveOverlay->getCenter()]->getSettings().mTimeIntervals.empty()) {
          return;
        }

        mTimeControl->setTimeSpeed(0);

        boost::posix_time::ptime time =
            cs::utils::convert::time::toPosix(mTimeControl->pSimulationTime.get());

        std::vector<TimeInterval> intervals =
            mActiveLayers[mActiveOverlay->getCenter()]->getSettings().mTimeIntervals;

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
            } else {
              // If the time is at the end of another interval, the next timestep is the
              // start time of the next interval.
              // Currently we trust that the intervals are ordered chronologically
              mTimeControl->setTime(cs::utils::convert::time::toSpice((it + 1)->mStartTime));
              return;
            }
          }
          // If the time was not the end time of any interval we can add the duration to
          // get the next timestep.
          sampleStartTime = utils::addDurationToTime(sampleStartTime, result.mSampleDuration);
          mTimeControl->setTime(cs::utils::convert::time::toSpice(sampleStartTime));
          return;
        }

        for (auto const& interval : intervals) {
          if (time < interval.mStartTime) {
            mTimeControl->setTime(cs::utils::convert::time::toSpice(interval.mStartTime));
            return;
          }
        }
      }));

  mGuiManager->getGui()->registerCallback(
      "wmsOverlays.goToLastTime", "Go to the last available timestep.", std::function([this]() {
        if (!mActiveOverlay || !mActiveLayers[mActiveOverlay->getCenter()] ||
            mActiveLayers[mActiveOverlay->getCenter()]->getSettings().mTimeIntervals.empty()) {
          return;
        }
        mTimeControl->setTimeSpeed(0);
        mTimeControl->setTime(
            cs::utils::convert::time::toSpice(mActiveLayers[mActiveOverlay->getCenter()]
                                                  ->getSettings()
                                                  .mTimeIntervals.back()
                                                  .mEndTime));
      }));

  mActiveBodyConnection = mSolarSystem->pActiveBody.connectAndTouch(
      [this](std::shared_ptr<cs::scene::CelestialBody> const& body) {
        if (!body) {
          return;
        }

        auto overlay = mWMSOverlays.find(body->getCenterName());

        mGuiManager->getGui()->callJavascript(
            "CosmoScout.sidebar.setTabEnabled", "WMS", overlay != mWMSOverlays.end());

        if (overlay == mWMSOverlays.end()) {
          mActiveOverlay = nullptr;
          return;
        }

        if (mActiveOverlay) {
          mActiveOverlay->pBounds.disconnect(mBoundsConnection);
        }
        mActiveOverlay = overlay->second;
        mActiveOverlay->pBounds.connect([this](Bounds bounds) {
          mGuiManager->getGui()->callJavascript("CosmoScout.wmsOverlays.setCurrentBounds",
              bounds.mMinLon, bounds.mMaxLon, bounds.mMinLat, bounds.mMaxLat);
        });

        mGuiManager->getGui()->callJavascript(
            "CosmoScout.gui.clearDropdown", "wmsOverlays.setServer");
        mGuiManager->getGui()->callJavascript(
            "CosmoScout.gui.addDropdownValue", "wmsOverlays.setServer", "None", "None", "false");

        resetWMSServer(overlay->second);

        auto const& settings = getBodySettings(overlay->second);
        for (auto const& server : mWms[body->getCenterName()]) {
          bool active = server.getTitle() == settings.mActiveServer.get();
          mGuiManager->getGui()->callJavascript("CosmoScout.gui.addDropdownValue",
              "wmsOverlays.setServer", server.getTitle(), server.getTitle(), active);

          if (active) {
            setWMSServer(overlay->second, server.getTitle());
          }
        }
      });

  mObserverSpeedConnection = mSolarSystem->pCurrentObserverSpeed.connect([this](float speed) {
    if (speed == 0.f) {
      mNoMovementSince           = std::chrono::high_resolution_clock::now();
      mNoMovement                = true;
      mNoMovementRequestedUpdate = false;
    } else {
      mNoMovement = false;
    }
  });

  onLoad();

  logger().info("Loading done.");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::deInit() {
  logger().info("Unloading plugin...");

  mSolarSystem->pActiveBody.disconnect(mActiveBodyConnection);
  mSolarSystem->pCurrentObserverSpeed.disconnect(mObserverSpeedConnection);

  mGuiManager->removePluginTab("WMS");
  mGuiManager->removeSettingsSection("WMS");

  mGuiManager->getGui()->callJavascript(
      "CosmoScout.gui.unregisterCss", "css/csp-simple-wms-bodies.css");

  mGuiManager->getGui()->unregisterCallback("wmsOverlays.setEnableTimeInterpolation");
  mGuiManager->getGui()->unregisterCallback("wmsOverlays.setEnableTimeSpan");
  mGuiManager->getGui()->unregisterCallback("wmsOverlays.setEnableAutomaticBoundsUpdate");

  mGuiManager->getGui()->unregisterCallback("wmsOverlays.setServer");
  mGuiManager->getGui()->unregisterCallback("wmsOverlays.setLayer");
  mGuiManager->getGui()->unregisterCallback("wmsOverlays.setStyle");

  mGuiManager->getGui()->unregisterCallback("wmsOverlays.goToFirstTime");
  mGuiManager->getGui()->unregisterCallback("wmsOverlays.goToPreviousTime");
  mGuiManager->getGui()->unregisterCallback("wmsOverlays.goToNextTime");
  mGuiManager->getGui()->unregisterCallback("wmsOverlays.goToLastTime");

  mGuiManager->getGui()->unregisterCallback("wmsOverlays.updateBounds");
  mGuiManager->getGui()->unregisterCallback("wmsOverlays.goToDefaultBounds");
  mGuiManager->getGui()->unregisterCallback("wmsOverlays.goToCurrentBounds");

  mAllSettings->onLoad().disconnect(mOnLoadConnection);
  mAllSettings->onSave().disconnect(mOnSaveConnection);

  logger().info("Unloading done.");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::update() {
  if (mPluginSettings->mEnableAutomaticBoundsUpdate.get() && mNoMovement &&
      !mNoMovementRequestedUpdate &&
      std::chrono::duration_cast<std::chrono::seconds>(
          std::chrono::high_resolution_clock::now() - mNoMovementSince)
              .count() > 2) {
    mNoMovementRequestedUpdate = true;

    if (mActiveOverlay) {
      mActiveOverlay->requestUpdateBounds();
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::onLoad() {
  // Read settings from JSON.
  from_json(mAllSettings->mPlugins.at("csp-wms-overlays"), *mPluginSettings);

  // First try to re-configure existing WMS overlays. We assume that they are similar if they
  // have the same name in the settings (which means they are attached to an anchor with the same
  // name).
  auto wmsOverlay = mWMSOverlays.begin();
  while (wmsOverlay != mWMSOverlays.end()) {
    auto settings = mPluginSettings->mBodies.find(wmsOverlay->first);
    if (settings != mPluginSettings->mBodies.end()) {
      // If there are settings for this simpleWMSBody, reconfigure it.
      wmsOverlay->second->configure(settings->second);

      setWMSServer(wmsOverlay->second, settings->second.mActiveServer.get());

      ++wmsOverlay;
    } else {
      // Else delete it.
      wmsOverlay = mWMSOverlays.erase(wmsOverlay);
    }
  }

  // Then add new WMS overlays.
  for (auto const& settings : mPluginSettings->mBodies) {
    if (mWMSOverlays.find(settings.first) != mWMSOverlays.end()) {
      continue;
    }

    auto anchor = mAllSettings->mAnchors.find(settings.first);

    if (anchor == mAllSettings->mAnchors.end()) {
      throw std::runtime_error(
          "There is no Anchor \"" + settings.first + "\" defined in the settings.");
    }

    auto wmsOverlay = std::make_shared<TextureOverlayRenderer>(
        settings.first, mSolarSystem, mTimeControl, mPluginSettings);

    mWMSOverlays.emplace(settings.first, wmsOverlay);

    for (auto const& wmsUrl : settings.second.mWms) {
      try {
        mWms[settings.first].emplace_back(wmsUrl, mPluginSettings->mCapabilityCache.get());
      } catch (std::exception const& e) {
        logger().warn("Failed to parse capabilities for '{}': {}", wmsUrl, e.what());
      }
    }

    if (!settings.second.mActiveServer.isDefault()) {
      setWMSServer(wmsOverlay, settings.second.mActiveServer.get());
    }
    wmsOverlay->configure(settings.second);
  }

  mSolarSystem->pActiveBody.touch(mActiveBodyConnection);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

Plugin::Settings::Body& Plugin::getBodySettings(
    std::shared_ptr<TextureOverlayRenderer> const& wmsOverlay) const {
  auto name = std::find_if(mWMSOverlays.begin(), mWMSOverlays.end(),
      [&](auto const& pair) { return pair.second == wmsOverlay; });
  return mPluginSettings->mBodies.at(name->first);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::setWMSServer(
    std::shared_ptr<TextureOverlayRenderer> const& wmsOverlay, std::string const& name) {

  resetWMSServer(wmsOverlay);

  auto&       settings = getBodySettings(wmsOverlay);
  auto const& server =
      std::find_if(mWms.at(wmsOverlay->getCenter()).begin(), mWms.at(wmsOverlay->getCenter()).end(),
          [&name](WebMapService wms) { return wms.getTitle() == name; });

  if (server == mWms.at(wmsOverlay->getCenter()).end()) {
    logger().warn("No server with name '{}' found", name);
    settings.mActiveServer.reset();

    mGuiManager->getGui()->callJavascript(
        "CosmoScout.gui.setDropdownValue", "wmsOverlays.setServer", "None", false);
    return;
  }

  settings.mActiveServer = name;
  mActiveServers[wmsOverlay->getCenter()].emplace(*server);

  WebMapLayer root = server->getRootLayer();
  for (auto const& layer : root.getAllLayers()) {
    addLayerToSelect(wmsOverlay, layer, settings.mActiveLayer.get());
  }
  mGuiManager->getGui()->callJavascript("CosmoScout.wmsOverlays.refreshLayerSelect");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::resetWMSServer(std::shared_ptr<TextureOverlayRenderer> const& wmsOverlay) {
  mGuiManager->getGui()->callJavascript("CosmoScout.wmsOverlays.resetLayerSelect");
  mGuiManager->getGui()->callJavascript("CosmoScout.gui.clearDropdown", "wmsOverlays.setLayer");
  mGuiManager->getGui()->callJavascript(
      "CosmoScout.gui.addDropdownValue", "wmsOverlays.setLayer", "None", "None", false);

  mActiveServers[wmsOverlay->getCenter()].reset();
  resetWMSLayer(wmsOverlay);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::setWMSLayer(
    std::shared_ptr<TextureOverlayRenderer> const& wmsOverlay, std::string const& name) {

  resetWMSLayer(wmsOverlay);

  auto& settings = getBodySettings(wmsOverlay);

  if (!mActiveServers[wmsOverlay->getCenter()]) {
    logger().warn("Can't set layer '{}': There is no active server for body '{}'", name,
        wmsOverlay->getCenter());
    settings.mActiveLayer.reset();
    mGuiManager->getGui()->callJavascript(
        "CosmoScout.gui.setDropdownValue", "wmsOverlays.setLayer", "None", false);
    return;
  }

  std::optional<WebMapLayer> layer = mActiveServers[wmsOverlay->getCenter()]->getLayer(name);

  if (!layer.has_value()) {
    logger().warn("Can't set layer '{}': No such layer found for server '{}'", name,
        mActiveServers[wmsOverlay->getCenter()]->getTitle());
    settings.mActiveLayer.reset();
    mGuiManager->getGui()->callJavascript(
        "CosmoScout.gui.setDropdownValue", "wmsOverlays.setLayer", "None", false);
    return;
  }

  settings.mActiveLayer = name;
  mActiveLayers[wmsOverlay->getCenter()].emplace(layer.value());
  wmsOverlay->setActiveWMS(mActiveServers[wmsOverlay->getCenter()].value(),
      mActiveLayers[wmsOverlay->getCenter()].value());

  mGuiManager->getGui()->callJavascript(
      "CosmoScout.wmsOverlays.setWMSDataCopyright", layer->getSettings().mAttribution.value_or(""));
  mGuiManager->getGui()->callJavascript("CosmoScout.wmsOverlays.setDefaultBounds",
      layer->getSettings().mBounds.mMinLon, layer->getSettings().mBounds.mMaxLon,
      layer->getSettings().mBounds.mMinLat, layer->getSettings().mBounds.mMaxLat);
  mGuiManager->getGui()->callJavascript(
      "CosmoScout.wmsOverlays.enableUpdateBounds", !layer->getSettings().mNoSubsets);
  mGuiManager->getGui()->callJavascript(
      "CosmoScout.wmsOverlays.enableTimeNavigation", !layer->getSettings().mTimeIntervals.empty());

  for (WebMapLayer::Style style : layer->getSettings().mStyles) {
    bool active = style.mName == settings.mActiveStyle.get();
    mGuiManager->getGui()->callJavascript("CosmoScout.gui.addDropdownValue", "wmsOverlays.setStyle",
        style.mName, style.mTitle, active);

    if (active) {
      setWMSStyle(wmsOverlay, style.mName);
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::resetWMSLayer(std::shared_ptr<TextureOverlayRenderer> const& wmsOverlay) {
  mGuiManager->getGui()->callJavascript("CosmoScout.wmsOverlays.setWMSDataCopyright", "");
  mGuiManager->getGui()->callJavascript("CosmoScout.wmsOverlays.clearDefaultBounds");
  mGuiManager->getGui()->callJavascript("CosmoScout.wmsOverlays.clearCurrentBounds");
  mGuiManager->getGui()->callJavascript("CosmoScout.wmsOverlays.enableUpdateBounds", false);
  mGuiManager->getGui()->callJavascript("CosmoScout.wmsOverlays.enableTimeNavigation", false);

  mActiveLayers[wmsOverlay->getCenter()].reset();
  wmsOverlay->clearActiveWMS();

  mGuiManager->getGui()->callJavascript("CosmoScout.gui.clearDropdown", "wmsOverlays.setStyle");
  mGuiManager->getGui()->callJavascript(
      "CosmoScout.gui.addDropdownValue", "wmsOverlays.setStyle", "", "Default", false);

  resetWMSStyle(wmsOverlay);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::setWMSStyle(
    std::shared_ptr<TextureOverlayRenderer> const& wmsOverlay, std::string const& name) {

  resetWMSStyle(wmsOverlay);

  auto                  bodySettings  = getBodySettings(wmsOverlay);
  WebMapLayer::Settings layerSettings = mActiveLayers[wmsOverlay->getCenter()]->getSettings();

  auto const& style = std::find_if(layerSettings.mStyles.begin(), layerSettings.mStyles.end(),
      [&name](WebMapLayer::Style style) { return style.mName == name; });
  if (style != layerSettings.mStyles.end()) {
    mGuiManager->getGui()->callJavascript(
        "CosmoScout.wmsOverlays.setLegendURL", style->mLegendUrl.value_or(""));
    bodySettings.mActiveStyle.set(name);
    wmsOverlay->setStyle(name);
  } else {
    wmsOverlay->setStyle("");
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::resetWMSStyle(std::shared_ptr<TextureOverlayRenderer> const& wmsOverlay) {
  mGuiManager->getGui()->callJavascript("CosmoScout.wmsOverlays.setLegendURL", "");

  wmsOverlay->setStyle("");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::addLayerToSelect(std::shared_ptr<TextureOverlayRenderer> const& wmsOverlay,
    WebMapLayer const& layer, std::string const& activeLayer, int const& depth) {
  bool active = layer.getName() == activeLayer;

  mGuiManager->getGui()->callJavascript("CosmoScout.wmsOverlays.addLayer", layer.getName(),
      layer.getTitle(), active, layer.isRequestable(), depth);

  if (active) {
    setWMSLayer(wmsOverlay, layer.getName());
  }

  for (auto const& sublayer : layer.getAllLayers()) {
    addLayerToSelect(wmsOverlay, sublayer, activeLayer, depth + 1);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::goToBounds(Bounds bounds) {
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

} // namespace csp::wmsoverlays
