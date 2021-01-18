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

  // Updates the bounds for which map data is requested.
  mGuiManager->getGui()->registerCallback(
      "wmsOverlays.updateBounds", "Updates the bounds for map requests.", std::function([this]() {
        auto overlay = mWMSOverlays.find(mSolarSystem->pActiveBody.get()->getCenterName());
        if (overlay != mWMSOverlays.end()) {
          overlay->second->requestUpdateBounds();
        }
      }));

  // Moves the observer to a position from which most of the current layer should be visible.
  mGuiManager->getGui()->registerCallback("wmsOverlays.goToDefaultBounds",
      "Fly the observer to the center of the default bounds of the current layer.",
      std::function([this]() {
        auto overlay = mWMSOverlays.find(mSolarSystem->pActiveBody.get()->getCenterName());
        if (overlay == mWMSOverlays.end()) {
          return;
        }
        auto        settings = getBodySettings(overlay->second);
        auto const& server   = std::find_if(mWms.at(overlay->second->getCenter()).begin(),
            mWms.at(overlay->second->getCenter()).end(), [&settings](WebMapService wms) {
              return wms.getTitle() == settings.mActiveServer.get();
            });
        if (server == mWms.at(overlay->second->getCenter()).end()) {
          return;
        }
        auto layer = server->getLayer(settings.mActiveLayer.get());
        if (!layer.has_value()) {
          return;
        }
        WebMapLayer::Settings layerSettings = layer->getSettings();

        double lon      = (layerSettings.mLonRange[0] + layerSettings.mLonRange[1]) / 2.;
        double lat      = (layerSettings.mLatRange[0] + layerSettings.mLatRange[1]) / 2.;
        double lonRange = layerSettings.mLonRange[1] - layerSettings.mLonRange[0];
        double latRange = layerSettings.mLatRange[1] - layerSettings.mLatRange[0];
        double maxRange = std::max(lonRange, latRange);
        // Rough approximation of the height, at which the whole bounds are in frame
        double fov    = 60.;
        double height = std::tan(cs::utils::convert::toRadians(maxRange) / 2.) *
                        mSolarSystem->getRadii(overlay->first)[0] /
                        std::tan(cs::utils::convert::toRadians(fov) / 2.);

        mSolarSystem->flyObserverTo(mSolarSystem->pActiveBody.get()->getCenterName(),
            mSolarSystem->pActiveBody.get()->getFrameName(),
            cs::utils::convert::toRadians(glm::dvec2(lon, lat)), height, 5.);
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
        auto overlay = mWMSOverlays.find(mSolarSystem->pActiveBody.get()->getCenterName());
        if (overlay != mWMSOverlays.end()) {
          setWMSServer(overlay->second, name);
          mNoMovementRequestedUpdate = false;
        }
      }));

  mGuiManager->getGui()->registerCallback("wmsOverlays.setLayer",
      "Set the current planet's WMS layer to the one with the given name.",
      std::function([this](std::string&& name) {
        auto overlay = mWMSOverlays.find(mSolarSystem->pActiveBody.get()->getCenterName());
        if (overlay != mWMSOverlays.end()) {
          setWMSLayer(overlay->second, name);
          mNoMovementRequestedUpdate = false;
        }
      }));

  mGuiManager->getGui()->registerCallback("wmsOverlays.setStyle",
      "Sets the style for the currently selected layer.", std::function([this](std::string&& name) {
        auto overlay = mWMSOverlays.find(mSolarSystem->pActiveBody.get()->getCenterName());
        if (overlay == mWMSOverlays.end()) {
          return;
        }
        auto        settings = getBodySettings(overlay->second);
        auto const& server   = std::find_if(mWms.at(overlay->second->getCenter()).begin(),
            mWms.at(overlay->second->getCenter()).end(), [&settings](WebMapService wms) {
              return wms.getTitle() == settings.mActiveServer.get();
            });
        if (server == mWms.at(overlay->second->getCenter()).end()) {
          return;
        }
        auto layer = server->getLayer(settings.mActiveLayer.get());
        if (!layer.has_value()) {
          return;
        }
        WebMapLayer::Settings layerSettings = layer->getSettings();
        auto const& style = std::find_if(layerSettings.mStyles.begin(), layerSettings.mStyles.end(),
            [&name](WebMapLayer::Style style) { return style.mName == name; });
        if (style != layerSettings.mStyles.end()) {
          mGuiManager->getGui()->callJavascript(
              "CosmoScout.wmsOverlays.setLegendURL", style->mLegendUrl.value_or(""));
          settings.mActiveStyle.set(name);
          overlay->second->setStyle(name);
        } else {
          mGuiManager->getGui()->callJavascript("CosmoScout.wmsOverlays.setLegendURL", "");
          settings.mActiveStyle.set("");
          overlay->second->setStyle("");
        }
        mNoMovementRequestedUpdate = false;
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
          return;
        }

        mGuiManager->getGui()->callJavascript(
            "CosmoScout.gui.clearDropdown", "wmsOverlays.setServer");
        mGuiManager->getGui()->callJavascript(
            "CosmoScout.gui.addDropdownValue", "wmsOverlays.setServer", "None", "None", "false");

        setWMSServer(overlay->second, "None");

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

  mSolarSystem->pCurrentObserverSpeed.connect([this](float speed) {
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

  mGuiManager->removePluginTab("WMS");
  mGuiManager->removeSettingsSection("WMS");

  mGuiManager->getGui()->callJavascript(
      "CosmoScout.gui.unregisterCss", "css/csp-simple-wms-bodies.css");

  mGuiManager->getGui()->unregisterCallback("wmsOverlays.setEnableTimeInterpolation");
  mGuiManager->getGui()->unregisterCallback("wmsOverlays.setEnableTimeSpan");
  mGuiManager->getGui()->unregisterCallback("wmsOverlays.setServer");
  mGuiManager->getGui()->unregisterCallback("wmsOverlays.setLayer");

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
    auto overlay = mWMSOverlays.find(mSolarSystem->pActiveBody.get()->getCenterName());

    if (overlay != mWMSOverlays.end()) {
      overlay->second->requestUpdateBounds();
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

    setWMSServer(wmsOverlay, settings.second.mActiveServer.get());
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
    std::shared_ptr<TextureOverlayRenderer> const& wmsOverlay, std::string const& name) const {
  mGuiManager->getGui()->callJavascript("CosmoScout.wmsOverlays.resetLayerSelect");
  mGuiManager->getGui()->callJavascript("CosmoScout.gui.clearDropdown", "wmsOverlays.setLayer");
  mGuiManager->getGui()->callJavascript(
      "CosmoScout.gui.addDropdownValue", "wmsOverlays.setLayer", "None", "None", false);

  auto&       settings = getBodySettings(wmsOverlay);
  auto const& server =
      std::find_if(mWms.at(wmsOverlay->getCenter()).begin(), mWms.at(wmsOverlay->getCenter()).end(),
          [&name](WebMapService wms) { return wms.getTitle() == name; });

  setWMSLayerNone(wmsOverlay);
  if (server == mWms.at(wmsOverlay->getCenter()).end()) {
    logger().trace("No server with name '{}' found", name);
    mGuiManager->getGui()->callJavascript(
        "CosmoScout.gui.setDropdownValue", "wmsOverlays.setServer", "None", false);
    return;
  }
  settings.mActiveServer = name;

  for (auto const& layer : server->getLayers()) {
    bool active = layer.getName() == settings.mActiveLayer.get();
    mGuiManager->getGui()->callJavascript("CosmoScout.gui.addDropdownValue", "wmsOverlays.setLayer",
        layer.getName(), layer.getTitle(), active);

    if (active) {
      setWMSLayer(wmsOverlay, *server, layer.getName());
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::setWMSLayer(
    std::shared_ptr<TextureOverlayRenderer> const& wmsOverlay, std::string const& name) const {
  auto&       settings = getBodySettings(wmsOverlay);
  auto const& server   = std::find_if(mWms.at(wmsOverlay->getCenter()).begin(),
      mWms.at(wmsOverlay->getCenter()).end(),
      [&settings](WebMapService wms) { return wms.getTitle() == settings.mActiveServer.get(); });

  if (server == mWms.at(wmsOverlay->getCenter()).end()) {
    setWMSLayerNone(wmsOverlay);
    return;
  }
  setWMSLayer(wmsOverlay, *server, name);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::setWMSLayer(std::shared_ptr<TextureOverlayRenderer> const& wmsOverlay,
    WebMapService const& server, std::string const& name) const {
  auto&                      settings = getBodySettings(wmsOverlay);
  std::optional<WebMapLayer> layer    = server.getLayer(name);

  if (!layer.has_value()) {
    logger().trace("No layer with name '{}' found", name);
    setWMSLayerNone(wmsOverlay);
    return;
  }
  settings.mActiveLayer = name;

  wmsOverlay->setActiveWMS(
      std::make_shared<WebMapService>(server), std::make_shared<WebMapLayer>(layer.value()));
  mGuiManager->getGui()->callJavascript(
      "CosmoScout.wmsOverlays.setWMSDataCopyright", layer->getSettings().mAttribution.value_or(""));
  mGuiManager->getGui()->callJavascript("CosmoScout.wmsOverlays.setDefaultBounds",
      layer->getSettings().mLonRange[0], layer->getSettings().mLonRange[1],
      layer->getSettings().mLatRange[0], layer->getSettings().mLatRange[1]);

  mGuiManager->getGui()->callJavascript("CosmoScout.gui.clearDropdown", "wmsOverlays.setStyle");
  mGuiManager->getGui()->callJavascript(
      "CosmoScout.gui.addDropdownValue", "wmsOverlays.setStyle", "", "Default", true);
  mGuiManager->getGui()->callJavascript("CosmoScout.wmsOverlays.setLegendURL", "");
  for (WebMapLayer::Style style : layer->getSettings().mStyles) {
    bool active = style.mName == settings.mActiveStyle.get();
    mGuiManager->getGui()->callJavascript("CosmoScout.gui.addDropdownValue", "wmsOverlays.setStyle",
        style.mName, style.mTitle, active);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::setWMSLayerNone(std::shared_ptr<TextureOverlayRenderer> const& wmsOverlay) const {
  auto& settings = getBodySettings(wmsOverlay);
  wmsOverlay->setActiveWMS(nullptr, nullptr);
  mGuiManager->getGui()->callJavascript(
      "CosmoScout.gui.setDropdownValue", "wmsOverlays.setLayer", "None", false);
  mGuiManager->getGui()->callJavascript("CosmoScout.wmsOverlays.setWMSDataCopyright", "");
  mGuiManager->getGui()->callJavascript("CosmoScout.wmsOverlays.clearDefaultBounds");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::wmsoverlays
