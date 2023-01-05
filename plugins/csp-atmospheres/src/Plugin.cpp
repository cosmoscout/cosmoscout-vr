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
    throw std::runtime_error(
        "Failed to parse Atmosphere::Model! Only 'CosmoScoutVR' or 'Bruneton' are allowed.");
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
  cs::core::Settings::deserialize(j, "enable", o.mEnable);
  cs::core::Settings::deserialize(j, "height", o.mHeight);
  cs::core::Settings::deserialize(j, "model", o.mModel);
  cs::core::Settings::deserialize(j, "modelSettings", o.mModelSettings);
  cs::core::Settings::deserialize(j, "enableWater", o.mEnableWater);
  cs::core::Settings::deserialize(j, "waterLevel", o.mWaterLevel);
  cs::core::Settings::deserialize(j, "enableClouds", o.mEnableClouds);
  cs::core::Settings::deserialize(j, "cloudTexture", o.mCloudTexture);
  cs::core::Settings::deserialize(j, "cloudAltitude", o.mCloudAltitude);
  cs::core::Settings::deserialize(j, "enableLightShafts", o.mEnableLightShafts);
}

void to_json(nlohmann::json& j, Plugin::Settings::Atmosphere const& o) {
  cs::core::Settings::serialize(j, "enable", o.mEnable);
  cs::core::Settings::serialize(j, "height", o.mHeight);
  cs::core::Settings::serialize(j, "model", o.mModel);
  cs::core::Settings::serialize(j, "modelSettings", o.mModelSettings);
  cs::core::Settings::serialize(j, "enableWater", o.mEnableWater);
  cs::core::Settings::serialize(j, "waterLevel", o.mWaterLevel);
  cs::core::Settings::serialize(j, "enableClouds", o.mEnableClouds);
  cs::core::Settings::serialize(j, "cloudTexture", o.mCloudTexture);
  cs::core::Settings::serialize(j, "cloudAltitude", o.mCloudAltitude);
  cs::core::Settings::serialize(j, "enableLightShafts", o.mEnableLightShafts);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void from_json(nlohmann::json const& j, Plugin::Settings& o) {
  cs::core::Settings::deserialize(j, "atmospheres", o.mAtmospheres);
}

void to_json(nlohmann::json& j, Plugin::Settings const& o) {
  cs::core::Settings::serialize(j, "atmospheres", o.mAtmospheres);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool Plugin::Settings::Atmosphere::operator==(Plugin::Settings::Atmosphere const& other) const {
  return mHeight == other.mHeight && mModel == other.mModel &&
         mModelSettings == other.mModelSettings && mEnable == other.mEnable &&
         mEnableWater == other.mEnableWater && mWaterLevel == other.mWaterLevel &&
         mEnableClouds == other.mEnableClouds && mCloudTexture == other.mCloudTexture &&
         mCloudAltitude == other.mCloudAltitude && mEnableLightShafts == other.mEnableLightShafts;
}

bool Plugin::Settings::Atmosphere::operator!=(Plugin::Settings::Atmosphere const& other) const {
  return !(*this == other);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::init() {

  logger().info("Loading plugin...");

  mOnLoadConnection = mAllSettings->onLoad().connect([this]() { onLoad(); });
  mOnSaveConnection = mAllSettings->onSave().connect([this]() { onSave(); });

  mGuiManager->addSettingsSectionToSideBarFromHTML(
      "Atmospheres", "blur_circular", "../share/resources/gui/atmospheres_settings.html");
  mGuiManager->executeJavascriptFile("../share/resources/gui/js/csp-atmospheres.js");

  mActiveObjectConnection = mSolarSystem->pActiveObject.connect(
      [this](std::shared_ptr<const cs::scene::CelestialObject> const& body) {
        mActiveAtmosphere = "";
        for (auto const& atmosphere : mAtmospheres) {
          if (body == mSolarSystem->getObject(atmosphere.first)) {
            mActiveAtmosphere = atmosphere.first;

            auto settings = mPluginSettings->mAtmospheres.at(atmosphere.first);
            mGuiManager->setCheckboxValue("atmosphere.setEnable", settings.mEnable.get());
            mGuiManager->setCheckboxValue("atmosphere.setEnableWater", settings.mEnableWater.get());
            mGuiManager->setCheckboxValue(
                "atmosphere.setEnableClouds", settings.mEnableClouds.get());
            mGuiManager->setCheckboxValue(
                "atmosphere.atmosphere.setEnableLightShafts", settings.mEnableLightShafts.get());
            mGuiManager->setSliderValue(
                "atmosphere.setCloudAltitude", settings.mCloudAltitude.get());
            mGuiManager->setSliderValue("atmosphere.setWaterLevel", settings.mWaterLevel.get());
          }
        }

        mGuiManager->getGui()->callJavascript(
            "CosmoScout.sidebar.setTabEnabled", "Atmospheres", mActiveAtmosphere != "");
      });

  mGuiManager->getGui()->registerCallback("atmosphere.setEnableWater",
      "Enables or disables rendering of a water surface.", std::function([this](bool enable) {
        if (mActiveAtmosphere != "") {
          auto& settings        = mPluginSettings->mAtmospheres.at(mActiveAtmosphere);
          settings.mEnableWater = enable;
          mAtmospheres.at(mActiveAtmosphere)->configure(settings);
        }
      }));

  mGuiManager->getGui()->registerCallback("atmosphere.setEnableClouds",
      "Enables or disables rendering of a cloud layer.", std::function([this](bool enable) {
        if (mActiveAtmosphere != "") {
          auto& settings         = mPluginSettings->mAtmospheres.at(mActiveAtmosphere);
          settings.mEnableClouds = enable;
          mAtmospheres.at(mActiveAtmosphere)->configure(settings);
        }
      }));

  mGuiManager->getGui()->registerCallback("atmosphere.setEnable",
      "Enables or disables rendering of atmospheres.", std::function([this](bool enable) {
        if (mActiveAtmosphere != "") {
          auto& settings   = mPluginSettings->mAtmospheres.at(mActiveAtmosphere);
          settings.mEnable = enable;
          mAtmospheres.at(mActiveAtmosphere)->configure(settings);
        }
      }));

  mGuiManager->getGui()->registerCallback("atmosphere.setEnableLightShafts",
      "If shadows are enabled, this enables or disables rendering of light shafts in the "
      "atmosphere.",
      std::function([this](bool enable) {
        if (mActiveAtmosphere != "") {
          auto& settings              = mPluginSettings->mAtmospheres.at(mActiveAtmosphere);
          settings.mEnableLightShafts = enable;
          mAtmospheres.at(mActiveAtmosphere)->configure(settings);
        }
      }));

  mGuiManager->getGui()->registerCallback("atmosphere.setCloudAltitude",
      "Higher values create a more realistic atmosphere.", std::function([this](double value) {
        if (mActiveAtmosphere != "") {
          auto& settings          = mPluginSettings->mAtmospheres.at(mActiveAtmosphere);
          settings.mCloudAltitude = static_cast<float>(value);
          mAtmospheres.at(mActiveAtmosphere)->configure(settings);
        }
      }));

  mGuiManager->getGui()->registerCallback("atmosphere.setWaterLevel",
      "Sets the height of the water surface relative to the planet's radius.",
      std::function([this](double value) {
        if (mActiveAtmosphere != "") {
          auto& settings       = mPluginSettings->mAtmospheres.at(mActiveAtmosphere);
          settings.mWaterLevel = static_cast<float>(value);
          mAtmospheres.at(mActiveAtmosphere)->configure(settings);
        }
      }));

  // mEnableShadowsConnection = mAllSettings->mGraphics.pEnableShadows.connect([this](bool value) {
  //   for (auto const& atmosphere : mAtmospheres) {
  //     if (value && mPluginSettings->mEnableLightShafts.get()) {
  //       atmosphere.second->getRenderer().setShadowMap(mGraphicsEngine->getShadowMap());
  //     } else {
  //       atmosphere.second->getRenderer().setShadowMap(nullptr);
  //     }
  //   }
  // });

  mEnableHDRConnection = mAllSettings->mGraphics.pEnableHDR.connect([this](bool val) {
    for (auto const& atmosphere : mAtmospheres) {
      if (val) {
        atmosphere.second->getRenderer().setHDRBuffer(mGraphicsEngine->getHDRBuffer());
      } else {
        atmosphere.second->getRenderer().setHDRBuffer(nullptr);
      }
    }
  });

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
  mGuiManager->getGui()->unregisterCallback("atmosphere.setEnableClouds");
  mGuiManager->getGui()->unregisterCallback("atmosphere.setEnableLightShafts");
  mGuiManager->getGui()->unregisterCallback("atmosphere.setCloudAltitude");
  mGuiManager->getGui()->unregisterCallback("atmosphere.setWaterLevel");

  mSolarSystem->pActiveObject.disconnect(mActiveObjectConnection);
  mAllSettings->mGraphics.pEnableShadows.disconnect(mEnableShadowsConnection);
  mAllSettings->mGraphics.pEnableHDR.disconnect(mEnableHDRConnection);
  mAllSettings->onLoad().disconnect(mOnLoadConnection);
  mAllSettings->onSave().disconnect(mOnSaveConnection);

  logger().info("Unloading done.");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::update() {
  for (auto const& atmosphere : mAtmospheres) {
    atmosphere.second->update();
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
    if (mAtmospheres.find(settings.first) != mAtmospheres.end()) {
      continue;
    }

    auto newAtmosphere =
        std::make_shared<Atmosphere>(mPluginSettings, mAllSettings, mSolarSystem, settings.first);
    newAtmosphere->configure(settings.second);

    mAtmospheres.emplace(settings.first, newAtmosphere);
  }

  mAllSettings->mGraphics.pEnableHDR.touch(mEnableHDRConnection);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::onSave() {
  mAllSettings->mPlugins["csp-atmospheres"] = *mPluginSettings;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::atmospheres
