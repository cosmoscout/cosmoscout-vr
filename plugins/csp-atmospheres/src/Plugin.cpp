////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "Plugin.hpp"

#include "../../../src/cs-core/GraphicsEngine.hpp"
#include "../../../src/cs-core/GuiManager.hpp"
#include "../../../src/cs-core/SolarSystem.hpp"
#include "../../../src/cs-utils/logger.hpp"
#include "Atmosphere.hpp"
#include "AtmosphereRenderer.hpp"
#include "logger.hpp"

#include <VistaKernel/GraphicsManager/VistaOpenGLNode.h>
#include <VistaKernel/GraphicsManager/VistaSceneGraph.h>
#include <VistaKernel/GraphicsManager/VistaTransformNode.h>

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

void from_json(nlohmann::json const& j, Plugin::Settings::Atmosphere& o) {
  cs::core::Settings::deserialize(j, "atmosphereHeight", o.mAtmosphereHeight);
  cs::core::Settings::deserialize(j, "mieHeight", o.mMieHeight);
  cs::core::Settings::deserialize(j, "mieScatteringR", o.mMieScatteringR);
  cs::core::Settings::deserialize(j, "mieScatteringG", o.mMieScatteringG);
  cs::core::Settings::deserialize(j, "mieScatteringB", o.mMieScatteringB);
  cs::core::Settings::deserialize(j, "mieAnisotropy", o.mMieAnisotropy);
  cs::core::Settings::deserialize(j, "rayleighHeight", o.mRayleighHeight);
  cs::core::Settings::deserialize(j, "rayleighScatteringR", o.mRayleighScatteringR);
  cs::core::Settings::deserialize(j, "rayleighScatteringG", o.mRayleighScatteringG);
  cs::core::Settings::deserialize(j, "rayleighScatteringB", o.mRayleighScatteringB);
  cs::core::Settings::deserialize(j, "rayleighAnisotropy", o.mRayleighAnisotropy);
  cs::core::Settings::deserialize(j, "cloudTexture", o.mCloudTexture);
  cs::core::Settings::deserialize(j, "cloudHeight", o.mCloudHeight);
}

void to_json(nlohmann::json& j, Plugin::Settings::Atmosphere const& o) {
  cs::core::Settings::serialize(j, "atmosphereHeight", o.mAtmosphereHeight);
  cs::core::Settings::serialize(j, "mieHeight", o.mMieHeight);
  cs::core::Settings::serialize(j, "mieScatteringR", o.mMieScatteringR);
  cs::core::Settings::serialize(j, "mieScatteringG", o.mMieScatteringG);
  cs::core::Settings::serialize(j, "mieScatteringB", o.mMieScatteringB);
  cs::core::Settings::serialize(j, "mieAnisotropy", o.mMieAnisotropy);
  cs::core::Settings::serialize(j, "rayleighHeight", o.mRayleighHeight);
  cs::core::Settings::serialize(j, "rayleighScatteringR", o.mRayleighScatteringR);
  cs::core::Settings::serialize(j, "rayleighScatteringG", o.mRayleighScatteringG);
  cs::core::Settings::serialize(j, "rayleighScatteringB", o.mRayleighScatteringB);
  cs::core::Settings::serialize(j, "rayleighAnisotropy", o.mRayleighAnisotropy);
  cs::core::Settings::serialize(j, "cloudTexture", o.mCloudTexture);
  cs::core::Settings::serialize(j, "cloudHeight", o.mCloudHeight);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void from_json(nlohmann::json const& j, Plugin::Settings& o) {
  cs::core::Settings::deserialize(j, "atmospheres", o.mAtmospheres);
  cs::core::Settings::deserialize(j, "enabled", o.mEnabled);
  cs::core::Settings::deserialize(j, "quality", o.mQuality);
  cs::core::Settings::deserialize(j, "waterLevel", o.mWaterLevel);
  cs::core::Settings::deserialize(j, "enableClouds", o.mEnableClouds);
  cs::core::Settings::deserialize(j, "enableLightShafts", o.mEnableLightShafts);
  cs::core::Settings::deserialize(j, "enableWater", o.mEnableWater);
}

void to_json(nlohmann::json& j, Plugin::Settings const& o) {
  cs::core::Settings::serialize(j, "atmospheres", o.mAtmospheres);
  cs::core::Settings::serialize(j, "enabled", o.mEnabled);
  cs::core::Settings::serialize(j, "quality", o.mQuality);
  cs::core::Settings::serialize(j, "waterLevel", o.mWaterLevel);
  cs::core::Settings::serialize(j, "enableClouds", o.mEnableClouds);
  cs::core::Settings::serialize(j, "enableLightShafts", o.mEnableLightShafts);
  cs::core::Settings::serialize(j, "enableWater", o.mEnableWater);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::init() {

  logger().info("Loading plugin...");

  mOnLoadConnection = mAllSettings->onLoad().connect([this]() { onLoad(); });
  mOnSaveConnection = mAllSettings->onSave().connect(
      [this]() { mAllSettings->mPlugins["csp-atmospheres"] = *mPluginSettings; });

  mGuiManager->addSettingsSectionToSideBarFromHTML(
      "Atmospheres", "blur_circular", "../share/resources/gui/atmospheres_settings.html");
  mGuiManager->addScriptToGuiFromJS("../share/resources/gui/js/csp-atmospheres.js");

  mGuiManager->getGui()->registerCallback("atmosphere.setEnableWater",
      "Enables or disables rendering of a water surface.",
      std::function([this](bool enable) { mPluginSettings->mEnableWater = enable; }));
  mPluginSettings->mEnableWater.connectAndTouch(
      [this](bool enable) { mGuiManager->setCheckboxValue("atmosphere.setEnableWater", enable); });

  mGuiManager->getGui()->registerCallback("atmosphere.setEnableClouds",
      "Enables or disables rendering of a cloud layer.",
      std::function([this](bool enable) { mPluginSettings->mEnableClouds = enable; }));
  mPluginSettings->mEnableClouds.connectAndTouch(
      [this](bool enable) { mGuiManager->setCheckboxValue("atmosphere.setEnableClouds", enable); });

  mGuiManager->getGui()->registerCallback("atmosphere.setEnable",
      "Enables or disables rendering of atmospheres.",
      std::function([this](bool enable) { mPluginSettings->mEnabled = enable; }));
  mPluginSettings->mEnabled.connectAndTouch(
      [this](bool enable) { mGuiManager->setCheckboxValue("atmosphere.setEnable", enable); });

  mGuiManager->getGui()->registerCallback("atmosphere.setEnableLightShafts",
      "If shadows are enabled, this enables or disables rendering of light shafts in the "
      "atmosphere.",
      std::function([this](bool enable) { mPluginSettings->mEnableLightShafts = enable; }));
  mPluginSettings->mEnableLightShafts.connectAndTouch([this](bool enable) {
    mGuiManager->setCheckboxValue("atmosphere.setEnableLightShafts", enable);
  });

  mGuiManager->getGui()->registerCallback("atmosphere.setQuality",
      "Higher values create a more realistic atmosphere.",
      std::function([this](double value) { mPluginSettings->mQuality = static_cast<int>(value); }));
  mPluginSettings->mQuality.connectAndTouch(
      [this](int value) { mGuiManager->setSliderValue("atmosphere.setQuality", value); });

  mGuiManager->getGui()->registerCallback("atmosphere.setWaterLevel",
      "Sets the height of the water surface relative to the planet's radius.",
      std::function(
          [this](double value) { mPluginSettings->mWaterLevel = static_cast<float>(value); }));
  mPluginSettings->mWaterLevel.connectAndTouch(
      [this](float value) { mGuiManager->setSliderValue("atmosphere.setWaterLevel", value); });

  mEnableShadowsConnection = mAllSettings->mGraphics.pEnableShadows.connect([this](bool value) {
    for (auto const& atmosphere : mAtmospheres) {
      if (value && mPluginSettings->mEnableLightShafts.get()) {
        atmosphere.second->getRenderer().setShadowMap(mGraphicsEngine->getShadowMap());
      } else {
        atmosphere.second->getRenderer().setShadowMap(nullptr);
      }
    }
  });

  mEnableHDRConnection = mAllSettings->mGraphics.pEnableHDR.connect([this](bool val) {
    for (auto const& atmosphere : mAtmospheres) {
      float const exposure = 0.6F;
      float const gamma    = 2.2F;
      atmosphere.second->getRenderer().setUseToneMapping(!val, exposure, gamma);
      if (val) {
        atmosphere.second->getRenderer().setHDRBuffer(mGraphicsEngine->getHDRBuffer());
      } else {
        atmosphere.second->getRenderer().setHDRBuffer(nullptr);
      }
    }
  });

  mAmbientBrightnessConnection =
      mAllSettings->mGraphics.pAmbientBrightness.connect([this](float val) {
        for (auto const& atmosphere : mAtmospheres) {
          float const ambientBrightnessModifier = 0.4F;
          atmosphere.second->getRenderer().setAmbientBrightness(val * ambientBrightnessModifier);
        }
      });

  mPluginSettings->mEnableLightShafts.connect([this](bool value) {
    for (auto const& atmosphere : mAtmospheres) {
      if (mAllSettings->mGraphics.pEnableShadows.get() && value) {
        atmosphere.second->getRenderer().setShadowMap(mGraphicsEngine->getShadowMap());
      } else {
        atmosphere.second->getRenderer().setShadowMap(nullptr);
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

  for (auto const& atmosphere : mAtmospheres) {
    mSolarSystem->unregisterAnchor(atmosphere.second);
  }

  mGuiManager->removeSettingsSection("Atmospheres");

  mGuiManager->getGui()->unregisterCallback("atmosphere.setEnableWater");
  mGuiManager->getGui()->unregisterCallback("atmosphere.setEnableClouds");
  mGuiManager->getGui()->unregisterCallback("atmosphere.setEnable");
  mGuiManager->getGui()->unregisterCallback("atmosphere.setEnableLightShafts");
  mGuiManager->getGui()->unregisterCallback("atmosphere.setQuality");
  mGuiManager->getGui()->unregisterCallback("atmosphere.setWaterLevel");

  mAllSettings->mGraphics.pEnableShadows.disconnect(mEnableShadowsConnection);
  mAllSettings->mGraphics.pEnableHDR.disconnect(mEnableHDRConnection);
  mAllSettings->mGraphics.pAmbientBrightness.disconnect(mAmbientBrightnessConnection);
  mAllSettings->onLoad().disconnect(mOnLoadConnection);
  mAllSettings->onSave().disconnect(mOnSaveConnection);

  logger().info("Unloading done.");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::update() {
  float fIntensity = 1.F;
  for (auto const& atmosphere : mAtmospheres) {
    if (mPluginSettings->mEnabled.get()) {
      float brightness = atmosphere.second->getRenderer().getApproximateSceneBrightness();
      fIntensity *= (1.F - brightness);
    }
  }

  mGraphicsEngine->pApproximateSceneBrightness = fIntensity;

  for (auto const& atmosphere : mAtmospheres) {
    double sunIlluminance = 10.0;

    if (mAllSettings->mGraphics.pEnableHDR.get()) {
      sunIlluminance = mSolarSystem->getSunIlluminance(atmosphere.second->getWorldTransform()[3]);
    }

    auto sunDirection = mSolarSystem->getSunDirection(atmosphere.second->getWorldTransform()[3]);

    atmosphere.second->getRenderer().setSun(sunDirection, static_cast<float>(sunIlluminance));
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
      mAllSettings->initAnchor(*atmosphere->second, settings->first);
      atmosphere->second->configure(settings->second);

      ++atmosphere;
    } else {
      // Else delete it.
      mSolarSystem->unregisterAnchor(atmosphere->second);
      atmosphere = mAtmospheres.erase(atmosphere);
    }
  }

  // Then add new atmospheres.
  for (auto const& settings : mPluginSettings->mAtmospheres) {
    if (mAtmospheres.find(settings.first) != mAtmospheres.end()) {
      continue;
    }

    auto atmosphere = std::make_shared<Atmosphere>(mPluginSettings, mAllSettings, settings.first);
    atmosphere->getRenderer().setHDRBuffer(mGraphicsEngine->getHDRBuffer());
    atmosphere->configure(settings.second);

    mSolarSystem->registerAnchor(atmosphere);

    mAtmospheres.emplace(settings.first, atmosphere);
  }

  mAllSettings->mGraphics.pEnableShadows.touch(mEnableShadowsConnection);
  mAllSettings->mGraphics.pEnableHDR.touch(mEnableHDRConnection);
  mAllSettings->mGraphics.pAmbientBrightness.touch(mAmbientBrightnessConnection);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::atmospheres
