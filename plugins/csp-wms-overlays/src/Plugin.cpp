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
  cs::core::Settings::deserialize(j, "activeWms", o.mActiveWMS);
  cs::core::Settings::deserialize(j, "wms", o.mWms);
}

void to_json(nlohmann::json& j, Plugin::Settings::Body const& o) {
  cs::core::Settings::serialize(j, "activeWms", o.mActiveWMS);
  cs::core::Settings::serialize(j, "wms", o.mWms);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void from_json(nlohmann::json const& j, Plugin::Settings& o) {
  cs::core::Settings::deserialize(j, "preFetch", o.mPrefetchCount);
  cs::core::Settings::deserialize(j, "mapCache", o.mMapCache);
  cs::core::Settings::deserialize(j, "bodies", o.mBodies);
}

void to_json(nlohmann::json& j, Plugin::Settings const& o) {
  cs::core::Settings::serialize(j, "preFetch", o.mPrefetchCount);
  cs::core::Settings::serialize(j, "mapCache", o.mMapCache);
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

  // Set whether to interpolate textures between timesteps (does not work when pre-fetch is
  // inactive).
  mGuiManager->getGui()->registerCallback("wmsOverlays.setEnableTimeInterpolation",
      "Enables or disables interpolation.",
      std::function([this](bool enable) { mPluginSettings->mEnableInterpolation = enable; }));

  // Set whether to display timespan.
  mGuiManager->getGui()->registerCallback("wmsOverlays.setEnableTimeSpan",
      "Enables or disables timespan.",
      std::function([this](bool enable) { mPluginSettings->mEnableTimespan = enable; }));

  // Set WMS source.
  mGuiManager->getGui()->registerCallback("wmsOverlays.setWMS",
      "Set the current planet's WMS source to the one with the given name.",
      std::function([this](std::string&& name) {
        auto overlay = mWMSOverlays.find(mSolarSystem->pActiveBody.get()->getCenterName());
        if (overlay != mWMSOverlays.end()) {
          setWMSSource(overlay->second, name);
        }
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
            "CosmoScout.gui.clearDropdown", "wmsOverlays.setWMS");
        mGuiManager->getGui()->callJavascript(
            "CosmoScout.gui.addDropdownValue", "wmsOverlays.setWMS", "None", "None", "false");

        auto const& settings = getBodySettings(overlay->second);
        for (auto const& layer : mWms[0].getLayers()) {
          bool active = layer.getName() == settings.mActiveWMS;
          mGuiManager->getGui()->callJavascript("CosmoScout.gui.addDropdownValue",
              "wmsOverlays.setWMS", layer.getName(), layer.getTitle(), active);
          if (active) {
            mGuiManager->getGui()->callJavascript("CosmoScout.wmsOverlays.setWMSDataCopyright",
                layer.getSettings().mAttribution.value_or(""));

            // Only allow setting timespan if it is specified for the WMS data set.
            mGuiManager->getGui()->callJavascript(
                "CosmoScout.wmsOverlays.enableCheckBox", true);
          }
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
  mGuiManager->getGui()->unregisterCallback("wmsOverlays.setWMS");

  mAllSettings->onLoad().disconnect(mOnLoadConnection);
  mAllSettings->onSave().disconnect(mOnSaveConnection);

  logger().info("Unloading done.");
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

      setWMSSource(wmsOverlay->second, settings->second.mActiveWMS);

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

    auto wmsOverlay =
        std::make_shared<TextureOverlayRenderer>(mSolarSystem, mTimeControl, mPluginSettings);

    mWMSOverlays.emplace(settings.first, wmsOverlay);

    for (auto const& wmsUrl : settings.second.mWms) {
      mWms.emplace_back(wmsUrl);
    }

    setWMSSource(wmsOverlay, settings.second.mActiveWMS);
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

void Plugin::setWMSSource(
    std::shared_ptr<TextureOverlayRenderer> const& wmsOverlay, std::string const& name) const {
  auto& settings = getBodySettings(wmsOverlay);

  if (name == "None") {
    wmsOverlay->setActiveWMS(nullptr, nullptr);
    mGuiManager->getGui()->callJavascript("CosmoScout.wmsOverlays.setWMSDataCopyright", "");
    settings.mActiveWMS = "None";
  } else {
    std::optional<WebMapLayer> layer = mWms[0].getLayer(name);

    if (!layer.has_value()) {
      logger().warn("Cannot set WMS layer '{}': There is no layer defined with this name! "
                    "Deselecting layer...",
          name);
      mGuiManager->getGui()->callJavascript(
          "CosmoScout.gui.setDropdownValue", "wmsOverlays.setWMS", "None", "true");
      return;
    }

    settings.mActiveWMS = name;

    wmsOverlay->setActiveWMS(
        std::make_shared<WebMapService>(mWms[0]), std::make_shared<WebMapLayer>(layer.value()));

    mGuiManager->getGui()->callJavascript("CosmoScout.wmsOverlays.setWMSDataCopyright",
        layer.value().getSettings().mAttribution.value_or(""));

    // Only allow setting timespan if it is specified for the WMS data set.
    mGuiManager->getGui()->callJavascript("CosmoScout.wmsOverlays.enableCheckBox", true);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::wmsoverlays
