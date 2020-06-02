////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "Plugin.hpp"

#include "DeepSpaceDot.hpp"
#include "SunFlare.hpp"
#include "Trajectory.hpp"
#include "logger.hpp"

#include "../../../src/cs-core/GuiManager.hpp"
#include "../../../src/cs-core/SolarSystem.hpp"
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
  mOnSaveConnection = mAllSettings->onSave().connect(
      [this]() { mAllSettings->mPlugins["csp-trajectories"] = *mPluginSettings; });

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

  for (auto const& flare : mSunFlares) {
    mSolarSystem->unregisterAnchor(flare.second);
  }

  for (auto const& trajectory : mTrajectories) {
    mSolarSystem->unregisterAnchor(trajectory.second);
  }

  for (auto const& dot : mDeepSpaceDots) {
    mSolarSystem->unregisterAnchor(dot.second);
  }

  mGuiManager->removeSettingsSection("Trajectories");

  mGuiManager->getGui()->unregisterCallback("trajectories.setEnableTrajectories");
  mGuiManager->getGui()->unregisterCallback("trajectories.setEnablePlanetMarks");
  mGuiManager->getGui()->unregisterCallback("trajectories.setEnableSunFlare");

  mAllSettings->onLoad().disconnect(mOnLoadConnection);
  mAllSettings->onSave().disconnect(mOnSaveConnection);

  logger().info("Unloading done.");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::onLoad() {

  // Read settings from JSON.
  from_json(mAllSettings->mPlugins.at("csp-trajectories"), *mPluginSettings);

  // We just recreate all SunFlares and DeepSpaceDots as they are quite cheap to construct. So
  // delete all existing ones first.
  for (auto const& flare : mSunFlares) {
    mSolarSystem->unregisterAnchor(flare.second);
  }
  for (auto const& dot : mDeepSpaceDots) {
    mSolarSystem->unregisterAnchor(dot.second);
  }
  mSunFlares.clear();
  mDeepSpaceDots.clear();

  // Now we go through all configured trajectories and create all required SunFlares and
  // DeepSpaceDots.
  for (auto const& settings : mPluginSettings->mTrajectories) {
    auto anchor = mAllSettings->mAnchors.find(settings.first);

    if (anchor == mAllSettings->mAnchors.end()) {
      logger().warn("Cannot add sun flare or planet mark for '{}': There is no such anchor defined "
                    "in the settings!",
          settings.first);
      continue;
    }

    auto [tStartExistence, tEndExistence] = anchor->second.getExistence();

    // Add the SunFlare.
    if (settings.second.mDrawFlare.value_or(false)) {
      auto flare = std::make_shared<SunFlare>(mAllSettings, mPluginSettings, anchor->second.mCenter,
          anchor->second.mFrame, tStartExistence, tEndExistence);
      mSolarSystem->registerAnchor(flare);

      flare->pColor =
          VistaColor(settings.second.mColor.r, settings.second.mColor.g, settings.second.mColor.b);

      mSunFlares[settings.first] = flare;
    }

    // Add the DeepSpaceDot.
    if (settings.second.mDrawDot.value_or(false)) {
      auto dot = std::make_shared<DeepSpaceDot>(mPluginSettings, anchor->second.mCenter,
          anchor->second.mFrame, tStartExistence, tEndExistence);
      mSolarSystem->registerAnchor(dot);

      dot->pColor =
          VistaColor(settings.second.mColor.r, settings.second.mColor.g, settings.second.mColor.b);

      // do not perform distance culling for DeepSpaceDots
      dot->pVisibleRadius = -1;
      dot->pVisible       = true;

      mDeepSpaceDots[settings.first] = dot;
    }
  }

  // For the trajectories we try to re-use as many as possible as they are quite expensive to
  // construct. First try to re-configure existing trajectories. A trajectory is re-used if it
  // shares the same target anchor name.
  for (auto trajectory = mTrajectories.begin(); trajectory != mTrajectories.end();) {
    auto settings = mPluginSettings->mTrajectories.find(trajectory->first);

    // If there are settings for this trajectory, reconfigure it.
    if (settings != mPluginSettings->mTrajectories.end() && settings->second.mTrail) {
      auto parentAnchor = mAllSettings->mAnchors.find(settings->second.mTrail->mParent);
      auto targetAnchor = mAllSettings->mAnchors.find(settings->first);

      // Ignore wrongly configured trajectories for now. The error will be emitted when we try to
      // add this as a new trajectory.
      if (parentAnchor != mAllSettings->mAnchors.end() &&
          targetAnchor != mAllSettings->mAnchors.end()) {

        auto [parentStartExistence, parentEndExistence] = parentAnchor->second.getExistence();
        auto [targetStartExistence, targetEndExistence] = targetAnchor->second.getExistence();
        trajectory->second->setStartExistence(std::max(parentStartExistence, targetStartExistence));
        trajectory->second->setEndExistence(std::min(parentEndExistence, targetEndExistence));
        trajectory->second->setCenterName(parentAnchor->second.mCenter);
        trajectory->second->setFrameName(parentAnchor->second.mFrame);
        trajectory->second->setTargetCenterName(targetAnchor->second.mCenter);
        trajectory->second->setTargetFrameName(targetAnchor->second.mFrame);
        trajectory->second->pSamples = settings->second.mTrail->mSamples;
        trajectory->second->pLength  = settings->second.mTrail->mLength;
        trajectory->second->pColor   = settings->second.mColor;

        ++trajectory;

        continue;
      }
    }

    // Else delete it.
    mSolarSystem->unregisterAnchor(trajectory->second);
    trajectory = mTrajectories.erase(trajectory);
  }

  // Then create all new trajectories.
  for (auto const& settings : mPluginSettings->mTrajectories) {
    if (settings.second.mTrail) {
      if (mTrajectories.find(settings.first) != mTrajectories.end()) {
        continue;
      }

      auto parentAnchor = mAllSettings->mAnchors.find(settings.second.mTrail->mParent);
      auto targetAnchor = mAllSettings->mAnchors.find(settings.first);

      if (parentAnchor == mAllSettings->mAnchors.end()) {
        logger().warn("Cannot add trajectory for '{}': There is no parent anchor '{}' defined in "
                      "the settings!",
            settings.first, settings.second.mTrail->mParent);
        continue;
      }

      if (targetAnchor == mAllSettings->mAnchors.end()) {
        logger().warn(
            "Cannot add trajectory for '{}': There is no such anchor defined in the settings!",
            settings.first);
        continue;
      }

      auto [parentStartExistence, parentEndExistence] = parentAnchor->second.getExistence();
      auto [targetStartExistence, targetEndExistence] = targetAnchor->second.getExistence();

      auto trajectory = std::make_shared<Trajectory>(mPluginSettings, targetAnchor->second.mCenter,
          targetAnchor->second.mFrame, parentAnchor->second.mCenter, parentAnchor->second.mFrame,
          std::max(parentStartExistence, targetStartExistence),
          std::min(parentEndExistence, targetEndExistence));

      trajectory->pSamples = settings.second.mTrail->mSamples;
      trajectory->pLength  = settings.second.mTrail->mLength;
      trajectory->pColor   = settings.second.mColor;

      // Change visibility of dots together with trajectory.
      trajectory->pVisible.connectAndTouch([this, anchorName = settings.first](bool visible) {
        auto dot = mDeepSpaceDots.find(anchorName);
        if (dot != mDeepSpaceDots.end()) {
          dot->second->pVisible = visible;
        }
      });

      mSolarSystem->registerAnchor(trajectory);

      mTrajectories.emplace(settings.first, trajectory);
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::trajectories
