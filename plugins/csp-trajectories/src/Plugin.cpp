////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "Plugin.hpp"

#include "DeepSpaceDot.hpp"
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
  cs::core::Settings::deserialize(j, "drawLDRFlare", o.mDrawLDRFlare);
  cs::core::Settings::deserialize(j, "drawHDRFlare", o.mDrawHDRFlare);
  cs::core::Settings::deserialize(j, "flareColor", o.mFlareColor);
  cs::core::Settings::deserialize(j, "ldrFlareScale", o.mLDRFlareScale);
  cs::core::Settings::deserialize(j, "trail", o.mTrail);
}

void to_json(nlohmann::json& j, Plugin::Settings::Trajectory const& o) {
  cs::core::Settings::serialize(j, "color", o.mColor);
  cs::core::Settings::serialize(j, "drawDot", o.mDrawDot);
  cs::core::Settings::serialize(j, "drawLDRFlare", o.mDrawLDRFlare);
  cs::core::Settings::serialize(j, "drawLDRFlare", o.mDrawLDRFlare);
  cs::core::Settings::serialize(j, "drawHDRFlare", o.mDrawHDRFlare);
  cs::core::Settings::serialize(j, "flareColor", o.mFlareColor);
  cs::core::Settings::serialize(j, "ldrFlareScale", o.mLDRFlareScale);
  cs::core::Settings::serialize(j, "trail", o.mTrail);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void from_json(nlohmann::json const& j, Plugin::Settings& o) {
  cs::core::Settings::deserialize(j, "trajectories", o.mTrajectories);
  cs::core::Settings::deserialize(j, "enableTrajectories", o.mEnableTrajectories);
  cs::core::Settings::deserialize(j, "enableLDRFlares", o.mEnableLDRFlares);
  cs::core::Settings::deserialize(j, "enableHDRFlares", o.mEnableHDRFlares);
  cs::core::Settings::deserialize(j, "enablePlanetMarks", o.mEnablePlanetMarks);
}

void to_json(nlohmann::json& j, Plugin::Settings const& o) {
  cs::core::Settings::serialize(j, "trajectories", o.mTrajectories);
  cs::core::Settings::serialize(j, "enableTrajectories", o.mEnableTrajectories);
  cs::core::Settings::serialize(j, "enableLDRFlares", o.mEnableLDRFlares);
  cs::core::Settings::serialize(j, "enableHDRFlares", o.mEnableHDRFlares);
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

  mGuiManager->getGui()->registerCallback("trajectories.setEnableTrajectoryDots",
      "Enables or disables the rendering of points marking the position of the planets.",
      std::function([this](bool value) { mPluginSettings->mEnablePlanetMarks = value; }));
  mPluginSettings->mEnablePlanetMarks.connectAndTouch([this](bool enable) {
    mGuiManager->setCheckboxValue("trajectories.setEnableTrajectoryDots", enable);
  });

  mGuiManager->getGui()->registerCallback("trajectories.setEnableLDRFlare",
      "Enables or disables the rendering of a glare around objects in non-HDR mode.",
      std::function([this](bool value) { mPluginSettings->mEnableLDRFlares = value; }));
  mPluginSettings->mEnableLDRFlares.connectAndTouch([this](bool enable) {
    mGuiManager->setCheckboxValue("trajectories.setEnableLDRFlare", enable);
  });

  mGuiManager->getGui()->registerCallback("trajectories.setEnableHDRFlare",
      "Enables or disables the rendering of a glare around objects in HDR mode.",
      std::function([this](bool value) { mPluginSettings->mEnableHDRFlares = value; }));
  mPluginSettings->mEnableHDRFlares.connectAndTouch([this](bool enable) {
    mGuiManager->setCheckboxValue("trajectories.setEnableHDRFlare", enable);
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
  mGuiManager->getGui()->unregisterCallback("trajectories.setEnableTrajectoryDots");
  mGuiManager->getGui()->unregisterCallback("trajectories.setEnableLDRFlare");

  mAllSettings->onLoad().disconnect(mOnLoadConnection);
  mAllSettings->onSave().disconnect(mOnSaveConnection);

  logger().info("Unloading done.");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::update() {
  for (auto const& trajectory : mTrajectories) {
    trajectory->update(mTimeControl->pSimulationTime.get());
  }

  auto updateVisibility = [this](std::unique_ptr<DeepSpaceDot> const& dot) {
    bool visible = false;

    switch (dot->pMode.get()) {
    case DeepSpaceDot::Mode::eMarker:
      visible = mPluginSettings->mEnablePlanetMarks.get();
      break;

    case DeepSpaceDot::Mode::eLDRFlare:
      visible =
          mPluginSettings->mEnableLDRFlares.get() && !mAllSettings->mGraphics.pEnableHDR.get();
      break;

    case DeepSpaceDot::Mode::eHDRFlare:
      visible = mPluginSettings->mEnableHDRFlares.get() && mAllSettings->mGraphics.pEnableHDR.get();
      break;

    default:
      break;
    }

    if (visible) {
      auto objectName = dot->getObjectName();
      auto object     = mSolarSystem->getObject(objectName);
      visible         = object && object->getIsOrbitVisible();
    }

    dot->pVisible = visible;

    return visible;
  };

  // If HDR is enabled, we draw a small dot at the position of the body. This will be always the
  // same size in screen space, no matter how far away the body is. This way we can always see the
  // position of the body, even if it is too small to see. We compute the luminance of the dot based
  // on the distance to the dot and the apparent phase angle of the body.
  for (auto const& flare : mHDRFlares) {
    if (updateVisibility(flare)) {
      auto objectName = flare->getObjectName();
      auto object     = mSolarSystem->getObject(objectName);

      auto   toBody   = object->getObserverRelativePosition();
      double bodyDist = glm::length(toBody);
      auto   toSun    = mSolarSystem->getSunDirection(object->getObserverRelativePosition());
      double sunDist  = glm::length(toSun);

      double phaseAngle = 2.0 * glm::asin(0.5 * glm::length(toBody / bodyDist - toSun / sunDist));
      double phase      = phaseAngle / glm::pi<double>();

      double sceneScale = mSolarSystem->getObserver().getScale();

      double bodyAngularSize = std::asin(object->getRadii()[0] / (bodyDist * sceneScale));
      double bodySolidAngle =
          4.0 * glm::pi<double>() * std::pow(std::sin(bodyAngularSize * 0.5), 2.0);
      flare->pSolidAngle = std::min(0.001, std::max(bodySolidAngle, 0.00001));
      double scaleFac    = bodySolidAngle / flare->pSolidAngle.get();

      double luminance = 0.0;

      if (objectName == "Sun") {
        luminance = scaleFac * mSolarSystem->getSunLuminance();

      } else {

        double illuminance = mSolarSystem->getSunIlluminance(object->getObserverRelativePosition());
        luminance          = phase * scaleFac * illuminance / glm::pi<double>();
      }

      flare->pLuminance = luminance;
    }
  }

  for (auto const& flare : mLDRFlares) {
    if (updateVisibility(flare)) {
      auto objectName = flare->getObjectName();
      auto object     = mSolarSystem->getObject(objectName);

      auto   toBody   = object->getObserverRelativePosition();
      double bodyDist = glm::length(toBody);

      double sceneScale = mSolarSystem->getObserver().getScale();

      double bodyAngularSize  = std::asin(object->getRadii()[0] / (bodyDist * sceneScale));
      double flareAngularSize = std::min(glm::pi<double>() * 0.5, bodyAngularSize * 10.0);

      flare->pSolidAngle =
          4.0 * glm::pi<double>() * std::pow(std::sin(flareAngularSize * 0.5), 2.0);

      logger().info("Body angle: {} Flare angle: {} Flare solid angle: {}",
          glm::degrees(bodyAngularSize), glm::degrees(flareAngularSize), flare->pSolidAngle.get());
    }
  }

  for (auto const& dot : mTrajectoryDots) {
    updateVisibility(dot);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::onLoad() {

  // Read settings from JSON.
  from_json(mAllSettings->mPlugins.at("csp-trajectories"), *mPluginSettings);

  size_t ldrFlareCount   = 0;
  size_t hdrFlareCount   = 0;
  size_t dotCount        = 0;
  size_t trajectoryCount = 0;

  for (auto const& settings : mPluginSettings->mTrajectories) {
    if (settings.second.mDrawLDRFlare.get()) {
      ++ldrFlareCount;
    }
    if (settings.second.mDrawHDRFlare.get()) {
      ++hdrFlareCount;
    }
    if (settings.second.mDrawDot.get()) {
      ++dotCount;
    }
    if (settings.second.mTrail) {
      ++trajectoryCount;
    }
  }

  // We just recreate all SunFlares and DeepSpaceDots as they are quite cheap to construct. So
  // delete all existing ones first.
  mLDRFlares.resize(ldrFlareCount);
  mHDRFlares.resize(hdrFlareCount);
  mTrajectoryDots.resize(dotCount);
  mTrajectories.resize(trajectoryCount);

  size_t ldrFlareIndex   = 0;
  size_t hdrFlareIndex   = 0;
  size_t dotIndex        = 0;
  size_t trajectoryIndex = 0;

  // Now we go through all configured trajectories and create all required SunFlares and
  // DeepSpaceDots.
  for (auto const& settings : mPluginSettings->mTrajectories) {

    // Add the non-HDR flare.
    if (settings.second.mDrawLDRFlare.get()) {
      if (!mLDRFlares[ldrFlareIndex]) {
        mLDRFlares[ldrFlareIndex] = std::make_unique<DeepSpaceDot>(mSolarSystem);
      }

      mLDRFlares[ldrFlareIndex]->setObjectName(settings.first);
      mLDRFlares[ldrFlareIndex]->pMode  = DeepSpaceDot::Mode::eLDRFlare;
      mLDRFlares[ldrFlareIndex]->pColor = VistaColor(settings.second.mFlareColor.get().r,
          settings.second.mFlareColor.get().g, settings.second.mFlareColor.get().b);

      ++ldrFlareIndex;
    }

    // Add the HDR flare.
    if (settings.second.mDrawHDRFlare.get()) {
      if (!mHDRFlares[hdrFlareIndex]) {
        mHDRFlares[hdrFlareIndex] = std::make_unique<DeepSpaceDot>(mSolarSystem);
      }

      mHDRFlares[hdrFlareIndex]->setObjectName(settings.first);
      mHDRFlares[hdrFlareIndex]->pMode  = DeepSpaceDot::Mode::eHDRFlare;
      mHDRFlares[hdrFlareIndex]->pColor = VistaColor(settings.second.mFlareColor.get().r,
          settings.second.mFlareColor.get().g, settings.second.mFlareColor.get().b);

      ++hdrFlareIndex;
    }

    // Add the DeepSpaceDot.
    if (settings.second.mDrawDot.get()) {
      if (!mTrajectoryDots[dotIndex]) {
        mTrajectoryDots[dotIndex] = std::make_unique<DeepSpaceDot>(mSolarSystem);
      }

      mTrajectoryDots[dotIndex]->setObjectName(settings.first);
      mTrajectoryDots[dotIndex]->pMode       = DeepSpaceDot::Mode::eMarker;
      mTrajectoryDots[dotIndex]->pSolidAngle = 0.00005F;
      mTrajectoryDots[dotIndex]->pColor =
          VistaColor(settings.second.mColor.r, settings.second.mColor.g, settings.second.mColor.b);
      mTrajectoryDots[dotIndex]->pVisible.connectFrom(mPluginSettings->mEnablePlanetMarks);

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
