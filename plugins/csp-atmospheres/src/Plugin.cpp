////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "Plugin.hpp"
#include "Atmosphere.hpp"

#include "../../../src/cs-core/GraphicsEngine.hpp"
#include "../../../src/cs-core/GuiManager.hpp"
#include "../../../src/cs-core/SolarSystem.hpp"
#include "../../../src/cs-core/TimeControl.hpp"
#include "../../../src/cs-utils/logger.hpp"
#include "logger.hpp"

////////////////////////////////////////////////////////////////////////////////////////////////////

EXPORT_FN cs::core::PluginBase* create() {
  return new csp::atmospheres::Plugin;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

EXPORT_FN void destroy(cs::core::PluginBase* pluginBase) {
  delete pluginBase; // NOLINT(cppcoreguidelines-owning-memory)
}

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace csp::atmospheres {

////////////////////////////////////////////////////////////////////////////////////////////////////

void from_json(nlohmann::json const& j, Plugin::Settings::Atmosphere::Model& o) {
  auto s = j.get<std::string>();
  if (s == "CosmoScoutVR") {
    o = Plugin::Settings::Atmosphere::Model::eCosmoScoutVR;
  } else if (s == "Bruneton") {
    o = Plugin::Settings::Atmosphere::Model::eBruneton;
  } else {
    throw std::runtime_error("Failed to parse Atmosphere::Model! Only 'CosmoScoutVR' and "
                             "'Bruneton' are allowed.");
  }
}

void to_json(nlohmann::json& j, Plugin::Settings::Atmosphere::Model o) {
  switch (o) {
  case Plugin::Settings::Atmosphere::Model::eCosmoScoutVR:
    j = "CosmoScoutVR";
    break;
  case Plugin::Settings::Atmosphere::Model::eBruneton:
    j = "Bruneton";
    break;
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void from_json(nlohmann::json const& j, Plugin::Settings::Atmosphere& o) {
  cs::core::Settings::deserialize(j, "topAltitude", o.mTopAltitude);
  cs::core::Settings::deserialize(j, "bottomAltitude", o.mBottomAltitude);
  cs::core::Settings::deserialize(j, "model", o.mModel);
  cs::core::Settings::deserialize(j, "modelSettings", o.mModelSettings);
  cs::core::Settings::deserialize(j, "enableWater", o.mEnableWater);
  cs::core::Settings::deserialize(j, "enableWaves", o.mEnableWaves);
  cs::core::Settings::deserialize(j, "waterLevel", o.mWaterLevel);
  cs::core::Settings::deserialize(j, "enableClouds", o.mEnableClouds);
  cs::core::Settings::deserialize(j, "cloudTexture", o.mCloudTexture);
  cs::core::Settings::deserialize(j, "cloudAltitude", o.mCloudAltitude);
  cs::core::Settings::deserialize(j, "renderSkydome", o.mRenderSkydome);
}

void to_json(nlohmann::json& j, Plugin::Settings::Atmosphere const& o) {
  cs::core::Settings::serialize(j, "topAltitude", o.mTopAltitude);
  cs::core::Settings::serialize(j, "bottomAltitude", o.mBottomAltitude);
  cs::core::Settings::serialize(j, "model", o.mModel);
  cs::core::Settings::serialize(j, "modelSettings", o.mModelSettings);
  cs::core::Settings::serialize(j, "enableWater", o.mEnableWater);
  cs::core::Settings::serialize(j, "enableWaves", o.mEnableWaves);
  cs::core::Settings::serialize(j, "waterLevel", o.mWaterLevel);
  cs::core::Settings::serialize(j, "enableClouds", o.mEnableClouds);
  cs::core::Settings::serialize(j, "cloudTexture", o.mCloudTexture);
  cs::core::Settings::serialize(j, "cloudAltitude", o.mCloudAltitude);
  cs::core::Settings::serialize(j, "renderSkydome", o.mRenderSkydome);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void from_json(nlohmann::json const& j, Plugin::Settings& o) {
  cs::core::Settings::deserialize(j, "atmospheres", o.mAtmospheres);
  cs::core::Settings::deserialize(j, "enable", o.mEnable);
}

void to_json(nlohmann::json& j, Plugin::Settings const& o) {
  cs::core::Settings::serialize(j, "atmospheres", o.mAtmospheres);
  cs::core::Settings::serialize(j, "enable", o.mEnable);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::init() {

  logger().info("Loading plugin...");

  mOnLoadConnection = mAllSettings->onLoad().connect([this]() { onLoad(); });
  mOnSaveConnection = mAllSettings->onSave().connect([this]() { onSave(); });

  mGuiManager->addSettingsSectionToSideBarFromHTML(
      "Atmospheres", "blur_circular", "../share/resources/gui/atmospheres_settings.html");
  mGuiManager->executeJavascriptFile("../share/resources/gui/js/csp-atmospheres.js");

  // Most settings of the sidebar are stored per-atmosphere. If the observer moves from one planet
  // to another, all sliders and checkboxes need to be updated to display the values of the newly
  // active body.
  mActiveObjectConnection = mSolarSystem->pActiveObject.connect(
      [this](std::shared_ptr<const cs::scene::CelestialObject> const& body) {
        mActiveAtmosphere = "";
        for (auto const& atmosphere : mAtmospheres) {
          if (body == mSolarSystem->getObject(atmosphere.first)) {
            mActiveAtmosphere = atmosphere.first;

            auto settings = mPluginSettings->mAtmospheres.at(atmosphere.first);
            mGuiManager->setCheckboxValue("atmosphere.setEnableWater", settings.mEnableWater.get());
            mGuiManager->setCheckboxValue("atmosphere.setEnableWaves", settings.mEnableWaves.get());
            mGuiManager->setSliderValue("atmosphere.setWaterLevel", settings.mWaterLevel.get());
            mGuiManager->setCheckboxValue(
                "atmosphere.setEnableClouds", settings.mEnableClouds.get());
            mGuiManager->setSliderValue(
                "atmosphere.setCloudAltitude", settings.mCloudAltitude.get());
          }
        }
      });

  mGuiManager->getGui()->registerCallback("atmosphere.setEnableWater",
      "Enables or disables rendering of a water surface.", std::function([this](bool enable) {
        if (!mActiveAtmosphere.empty()) {
          auto& settings        = mPluginSettings->mAtmospheres.at(mActiveAtmosphere);
          settings.mEnableWater = enable;
          mAtmospheres.at(mActiveAtmosphere)->configure(settings);
        }
      }));

  mGuiManager->getGui()->registerCallback("atmosphere.setEnableWaves",
      "Enables or disables rendering of waves on the water surface.",
      std::function([this](bool enable) {
        if (!mActiveAtmosphere.empty()) {
          auto& settings        = mPluginSettings->mAtmospheres.at(mActiveAtmosphere);
          settings.mEnableWaves = enable;
          mAtmospheres.at(mActiveAtmosphere)->configure(settings);
        }
      }));

  mGuiManager->getGui()->registerCallback("atmosphere.setWaterLevel",
      "Sets the height of the water surface in meters.", std::function([this](double value) {
        if (!mActiveAtmosphere.empty()) {
          auto& settings       = mPluginSettings->mAtmospheres.at(mActiveAtmosphere);
          settings.mWaterLevel = static_cast<float>(value);
          mAtmospheres.at(mActiveAtmosphere)->configure(settings);
        }
      }));

  mGuiManager->getGui()->registerCallback("atmosphere.setEnableClouds",
      "Enables or disables rendering of a cloud layer.", std::function([this](bool enable) {
        if (!mActiveAtmosphere.empty()) {
          auto& settings         = mPluginSettings->mAtmospheres.at(mActiveAtmosphere);
          settings.mEnableClouds = enable;
          mAtmospheres.at(mActiveAtmosphere)->configure(settings);
        }
      }));

  mGuiManager->getGui()->registerCallback("atmosphere.setCloudAltitude",
      "Higher values create a more realistic atmosphere.", std::function([this](double value) {
        if (!mActiveAtmosphere.empty()) {
          auto& settings          = mPluginSettings->mAtmospheres.at(mActiveAtmosphere);
          settings.mCloudAltitude = static_cast<float>(value);
          mAtmospheres.at(mActiveAtmosphere)->configure(settings);
        }
      }));

  mGuiManager->getGui()->registerCallback("atmosphere.setEnable",
      "Enables or disables rendering of atmospheres.",
      std::function([this](bool enable) { mPluginSettings->mEnable = enable; }));

  // Load settings.
  onLoad();

  logger().info("Loading done.");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::deInit() {
  logger().info("Unloading plugin...");

  // Save settings as this plugin may get reloaded.
  onSave();

  mGuiManager->removeSettingsSection("Atmospheres");

  mGuiManager->getGui()->callJavascript("CosmoScout.removeApi", "atmosphere");

  mGuiManager->getGui()->unregisterCallback("atmosphere.setEnable");
  mGuiManager->getGui()->unregisterCallback("atmosphere.setEnableWater");
  mGuiManager->getGui()->unregisterCallback("atmosphere.setWaterLevel");
  mGuiManager->getGui()->unregisterCallback("atmosphere.setEnableClouds");
  mGuiManager->getGui()->unregisterCallback("atmosphere.setCloudAltitude");

  mSolarSystem->pActiveObject.disconnect(mActiveObjectConnection);
  mAllSettings->onLoad().disconnect(mOnLoadConnection);
  mAllSettings->onSave().disconnect(mOnSaveConnection);

  logger().info("Unloading done.");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::update() {
  for (auto const& atmosphere : mAtmospheres) {
    atmosphere.second->update(mTimeControl->pSimulationTime.get());
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::onLoad() {
  // Read settings from JSON.
  from_json(mAllSettings->mPlugins.at("csp-atmospheres"), *mPluginSettings);

  // First try to re-configure existing atmospheres. We assume that they are similar if they have
  // the same name in the settings (which means they are attached to an anchor with the same name).
  auto atmosphere = mAtmospheres.begin();
  while (atmosphere != mAtmospheres.end()) {
    auto settings = mPluginSettings->mAtmospheres.find(atmosphere->first);
    if (settings != mPluginSettings->mAtmospheres.end()) {
      // If there are settings for this atmosphere, reconfigure it.
      atmosphere->second->configure(settings->second);

      ++atmosphere;
    } else {
      // Else delete it.
      atmosphere = mAtmospheres.erase(atmosphere);
    }
  }

  // Then add new atmospheres.
  for (auto const& settings : mPluginSettings->mAtmospheres) {

    // We already have created that atmosphere.
    if (mAtmospheres.find(settings.first) != mAtmospheres.end()) {
      continue;
    }

    auto newAtmosphere = std::make_shared<Atmosphere>(
        mPluginSettings, mAllSettings, mSolarSystem, mGraphicsEngine, settings.first);
    newAtmosphere->configure(settings.second);

    mAtmospheres.emplace(settings.first, newAtmosphere);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::onSave() {
  mAllSettings->mPlugins["csp-atmospheres"] = *mPluginSettings;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::atmospheres
