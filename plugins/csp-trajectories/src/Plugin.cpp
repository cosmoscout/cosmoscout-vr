////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "Plugin.hpp"

#include "../../../src/cs-core/DeepSpaceDot.hpp"
#include "SunFlare.hpp"
#include "Trajectory.hpp"
#include "logger.hpp"

#include "../../../src/cs-core/GuiManager.hpp"
#include "../../../src/cs-core/SolarSystem.hpp"
#include "../../../src/cs-core/TimeControl.hpp"
#include "../../../src/cs-utils/logger.hpp"

////////////////////////////////////////////////////////////////////////////////////////////////////

EXPORT_FN cs::core::PluginBase* create() {
  return new csp::trajectories::Plugin;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

EXPORT_FN void destroy(cs::core::PluginBase* pluginBase) {
  delete pluginBase; // NOLINT(cppcoreguidelines-owning-memory)
}

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace csp::trajectories {

////////////////////////////////////////////////////////////////////////////////////////////////////

void from_json(nlohmann::json const& j, Plugin::Settings::Trajectory::Trail& o) {
  cs::core::Settings::deserialize(j, "length", o.mLength);
  cs::core::Settings::deserialize(j, "samples", o.mSamples);
  cs::core::Settings::deserialize(j, "parent", o.mParent);
}

void to_json(nlohmann::json& j, Plugin::Settings::Trajectory::Trail const& o) {
  cs::core::Settings::serialize(j, "length", o.mLength);
  cs::core::Settings::serialize(j, "samples", o.mSamples);
  cs::core::Settings::serialize(j, "parent", o.mParent);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void from_json(nlohmann::json const& j, Plugin::Settings::Trajectory& o) {
  cs::core::Settings::deserialize(j, "color", o.mColor);
  cs::core::Settings::deserialize(j, "drawDot", o.mDrawDot);
  cs::core::Settings::deserialize(j, "drawFlare", o.mDrawFlare);
  cs::core::Settings::deserialize(j, "trail", o.mTrail);
}

void to_json(nlohmann::json& j, Plugin::Settings::Trajectory const& o) {
  cs::core::Settings::serialize(j, "color", o.mColor);
  cs::core::Settings::serialize(j, "drawDot", o.mDrawDot);
  cs::core::Settings::serialize(j, "drawFlare", o.mDrawFlare);
  cs::core::Settings::serialize(j, "trail", o.mTrail);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void from_json(nlohmann::json const& j, Plugin::Settings& o) {
  cs::core::Settings::deserialize(j, "trajectories", o.mTrajectories);
  cs::core::Settings::deserialize(j, "enableTrajectories", o.mEnableTrajectories);
  cs::core::Settings::deserialize(j, "enableSunFlares", o.mEnableSunFlares);
  cs::core::Settings::deserialize(j, "enablePlanetMarks", o.mEnablePlanetMarks);
}

void to_json(nlohmann::json& j, Plugin::Settings const& o) {
  cs::core::Settings::serialize(j, "trajectories", o.mTrajectories);
  cs::core::Settings::serialize(j, "enableTrajectories", o.mEnableTrajectories);
  cs::core::Settings::serialize(j, "enableSunFlares", o.mEnableSunFlares);
  cs::core::Settings::serialize(j, "enablePlanetMarks", o.mEnablePlanetMarks);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::init() {

  logger().info("Loading plugin...");

  mOnLoadConnection = mAllSettings->onLoad().connect([this]() { onLoad(); });
  mOnSaveConnection = mAllSettings->onSave().connect([this]() { onSave(); });

  mGuiManager->addSettingsSectionToSideBarFromHTML("Trajectories", "radio_button_unchecked",
      "../share/resources/gui/trajectories-settings.html");

  mGuiManager->getGui()->registerCallback("trajectories.setEnableTrajectories",
      "Enables or disables the rendering of trajectories.",
      std::function([this](bool value) { mPluginSettings->mEnableTrajectories = value; }));
  mPluginSettings->mEnableTrajectories.connectAndTouch([this](bool enable) {
    mGuiManager->setCheckboxValue("trajectories.setEnableTrajectories", enable);
  });

  mGuiManager->getGui()->registerCallback("trajectories.setEnablePlanetMarks",
      "Enables or disables the rendering of points marking the position of the planets.",
      std::function([this](bool value) { mPluginSettings->mEnablePlanetMarks = value; }));
  mPluginSettings->mEnablePlanetMarks.connectAndTouch([this](bool enable) {
    mGuiManager->setCheckboxValue("trajectories.setEnablePlanetMarks", enable);
  });

  mGuiManager->getGui()->registerCallback("trajectories.setEnableSunFlare",
      "Enables or disables the rendering of a glare around the sun.",
      std::function([this](bool value) { mPluginSettings->mEnableSunFlares = value; }));
  mPluginSettings->mEnableSunFlares.connectAndTouch([this](bool enable) {
    mGuiManager->setCheckboxValue("trajectories.setEnableSunFlare", enable);
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

  mGuiManager->removeSettingsSection("Trajectories");

  mGuiManager->getGui()->unregisterCallback("trajectories.setEnableTrajectories");
  mGuiManager->getGui()->unregisterCallback("trajectories.setEnablePlanetMarks");
  mGuiManager->getGui()->unregisterCallback("trajectories.setEnableSunFlare");

  mAllSettings->onLoad().disconnect(mOnLoadConnection);
  mAllSettings->onSave().disconnect(mOnSaveConnection);

  logger().info("Unloading done.");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::update() {
  for (auto const& trajectory : mTrajectories) {
    trajectory->update(mTimeControl->pSimulationTime.get());
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::onLoad() {

  // Read settings from JSON.
  from_json(mAllSettings->mPlugins.at("csp-trajectories"), *mPluginSettings);

  size_t flareCount      = 0;
  size_t dotCount        = 0;
  size_t trajectoryCount = 0;

  for (auto const& settings : mPluginSettings->mTrajectories) {
    if (settings.second.mDrawFlare.value_or(false)) {
      ++flareCount;
    }
    if (settings.second.mDrawDot.value_or(false)) {
      ++dotCount;
    }
    if (settings.second.mTrail) {
      ++trajectoryCount;
    }
  }

  // We just recreate all SunFlares and DeepSpaceDots as they are quite cheap to construct. So
  // delete all existing ones first.
  mSunFlares.resize(flareCount);
  mDeepSpaceDots.resize(dotCount);
  mTrajectories.resize(trajectoryCount);

  size_t flareIndex      = 0;
  size_t dotIndex        = 0;
  size_t trajectoryIndex = 0;

  // Now we go through all configured trajectories and create all required SunFlares and
  // DeepSpaceDots.
  for (auto const& settings : mPluginSettings->mTrajectories) {

    // Add the SunFlare.
    if (settings.second.mDrawFlare.value_or(false)) {
      if (!mSunFlares[flareIndex]) {
        mSunFlares[flareIndex] =
            std::make_unique<SunFlare>(mAllSettings, mPluginSettings, mSolarSystem);
      }

      mSunFlares[flareIndex]->setObjectName(settings.first);
      mSunFlares[flareIndex]->pColor =
          VistaColor(settings.second.mColor.r, settings.second.mColor.g, settings.second.mColor.b);

      ++flareIndex;
    }

    // Add the DeepSpaceDot.
    if (settings.second.mDrawDot.value_or(false)) {
      if (!mDeepSpaceDots[dotIndex]) {
        mDeepSpaceDots[dotIndex] = std::make_unique<cs::core::DeepSpaceDot>(mSolarSystem);
      }

      mDeepSpaceDots[dotIndex]->setObjectName(settings.first);
      mDeepSpaceDots[dotIndex]->pColor =
          VistaColor(settings.second.mColor.r, settings.second.mColor.g, settings.second.mColor.b);
      mDeepSpaceDots[dotIndex]->pVisible.connectFrom(mPluginSettings->mEnablePlanetMarks);

      ++dotIndex;
    }

    // Then create all new trajectories.
    if (settings.second.mTrail) {
      if (!mTrajectories[trajectoryIndex]) {
        mTrajectories[trajectoryIndex] =
            std::make_unique<Trajectory>(mPluginSettings, mSolarSystem);
      }
      auto targetAnchor = settings.first;
      auto parentAnchor = settings.second.mTrail->mParent;

      mTrajectories[trajectoryIndex]->setTargetName(targetAnchor);
      mTrajectories[trajectoryIndex]->setParentName(parentAnchor);
      mTrajectories[trajectoryIndex]->pSamples = settings.second.mTrail->mSamples;
      mTrajectories[trajectoryIndex]->pLength  = settings.second.mTrail->mLength;
      mTrajectories[trajectoryIndex]->pColor   = settings.second.mColor;

      ++trajectoryIndex;
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::onSave() {
  mAllSettings->mPlugins["csp-trajectories"] = *mPluginSettings;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::trajectories
