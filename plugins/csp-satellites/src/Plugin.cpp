////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "Plugin.hpp"

#include "Satellite.hpp"
#include "logger.hpp"

#include "../../../src/cs-core/GuiManager.hpp"
#include "../../../src/cs-core/SolarSystem.hpp"
#include "../../../src/cs-utils/logger.hpp"

#include <cspice/SpiceUsr.h>

////////////////////////////////////////////////////////////////////////////////////////////////////

EXPORT_FN cs::core::PluginBase* create() {
  return new csp::satellites::Plugin;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

EXPORT_FN void destroy(cs::core::PluginBase* pluginBase) {
  delete pluginBase; // NOLINT(cppcoreguidelines-owning-memory)
}

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace csp::satellites {

////////////////////////////////////////////////////////////////////////////////////////////////////

void from_json(nlohmann::json const& j, Plugin::Settings::Satellite& o) {
  cs::core::Settings::deserialize(j, "modelFile", o.mModelFile);
  cs::core::Settings::deserialize(j, "environmentMap", o.mEnvironmentMap);
  cs::core::Settings::deserialize(j, "fieldOfView", o.mFieldOfView);
  cs::core::Settings::deserialize(j, "cameraObject", o.mCameraObject);
}

void to_json(nlohmann::json& j, Plugin::Settings::Satellite const& o) {
  cs::core::Settings::serialize(j, "modelFile", o.mModelFile);
  cs::core::Settings::serialize(j, "environmentMap", o.mEnvironmentMap);
  cs::core::Settings::serialize(j, "fieldOfView", o.mFieldOfView);
  cs::core::Settings::serialize(j, "cameraObject", o.mCameraObject);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void from_json(nlohmann::json const& j, Plugin::Settings& o) {
  cs::core::Settings::deserialize(j, "satellites", o.mSatellites);
}

void to_json(nlohmann::json& j, Plugin::Settings const& o) {
  cs::core::Settings::serialize(j, "satellites", o.mSatellites);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void from_json(nlohmann::json const& j, Plugin::ExtraSatellite& o) {
  cs::core::Settings::deserialize(j, "bodyName", o.bodyName);
  cs::core::Settings::deserialize(j, "bodyId", o.bodyId);
  cs::core::Settings::deserialize(j, "posJobId", o.posJobId);
  cs::core::Settings::deserialize(j, "orientJobId", o.orientJobId);
  cs::core::Settings::deserialize(j, "existenceStart", o.existenceStart);
  cs::core::Settings::deserialize(j, "existenceEnd", o.existenceEnd);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

Plugin::Plugin()
    : mDownloader(4) {
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::init() {

  logger().info("Loading plugin...");

  mOnLoadConnection = mAllSettings->onLoad().connect([this]() { onLoad(); });
  mOnSaveConnection = mAllSettings->onSave().connect([this]() { onSave(); });

  mGuiManager->addTemplate(
      "satellite-view-template", "../share/resources/gui/csp-satellites-template.html");
  mGuiManager->addPluginTabToSideBarFromHTML(
      "Satellites", "satellite", "../share/resources/gui/csp-satellites-tab.html");
  mGuiManager->executeJavascriptFile("../share/resources/gui/js/csp-satellites.js");

  mGuiManager->getGui()->registerCallback("satellites.addSatellite",
      "Succesfully requested data for a new satellite, now load it into the plugin.",
      std::function([this](std::string json) {
        ExtraSatellite satellite = nlohmann::json::parse(json);
        downloadSatelliteKernels(std::move(satellite));
      }));
  mGuiManager->getGui()->registerCallback("satellites.setFieldOfView",
      "Set the field of view for the given satellite.",
      std::function([this](std::string satellite, double fov) {
        mPluginSettings.mSatellites[satellite].mFieldOfView = fov;
      }));
  mGuiManager->getGui()->registerCallback("satellites.setSatelliteModel",
      "Set the model to be shown for a satellite.", std::function([this](std::string modelFile) {
        for (auto& satelliteSetting : mPluginSettings.mSatellites) {
          satelliteSetting.second.mModelFile.set(modelFile);
        }
      }));

  // Load settings.
  onLoad();

  logger().info("Loading done.");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::deInit() {
  logger().info("Unloading plugin...");

  // Save settings as this plugin may get reloaded.
  onSave();

  mSatellites.clear();

  mAllSettings->onLoad().disconnect(mOnLoadConnection);
  mAllSettings->onSave().disconnect(mOnSaveConnection);

  logger().info("Unloading done.");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::update() {
  for (auto const& satellite : mSatellites) {
    satellite->update();
  }
  if (!mPendingDownloads.empty() && mDownloader.hasFinished()) {
    loadSatelliteKernel();
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::onLoad() {

  mSatellites.clear();

  // Read settings from JSON.
  mPluginSettings = mAllSettings->mPlugins.at("csp-satellites");

  for (auto const& settings : mPluginSettings.mSatellites) {
    addSatellite(settings.first, settings.second);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::onSave() {
  mAllSettings->mPlugins["csp-satellites"] = mPluginSettings;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::string Plugin::downloadKernel(
    std::string const& jobId, std::string const& type, std::string const& extension) {
  std::stringstream downloadPath;
  downloadPath << "http://localhost:8000/jobs/" << jobId << "/result/" << type;
  std::stringstream localPath;
  localPath << "./spice_out/" << jobId << extension;
  mDownloader.download(downloadPath.str(), localPath.str());
  return localPath.str();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::downloadSatelliteKernels(ExtraSatellite&& satellite) {
  satellite.kernelPaths["SPK"] = downloadKernel(satellite.posJobId, "bsp", ".bsp");
  satellite.kernelPaths["CK"] = downloadKernel(satellite.orientJobId, "ck", ".bck");
  satellite.kernelPaths["SCLK"] = downloadKernel(satellite.orientJobId, "tc", ".tsc");
  satellite.kernelPaths["FK"]   = downloadKernel(satellite.orientJobId, "fk", ".tf");
  mPendingDownloads.push_back(satellite);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::loadSatelliteKernel() {
  bool failed = false;
  for (ExtraSatellite const& sat : mPendingDownloads) {
    // Load the spice kernels.
    furnsh_c(sat.kernelPaths.at("SPK").c_str());
    furnsh_c(sat.kernelPaths.at("CK").c_str());
    furnsh_c(sat.kernelPaths.at("SCLK").c_str());
    furnsh_c(sat.kernelPaths.at("FK").c_str());

    if (failed_c()) {
      int32_t const maxSpiceErrorLength = 320;

      std::array<SpiceChar, maxSpiceErrorLength> msg{};
      getmsg_c("LONG", maxSpiceErrorLength, msg.data());
      logger().error(msg.data());
      failed = true;
    }

    if (mAllSettings->mObjects.find(sat.bodyName) == mAllSettings->mObjects.end()) {
      std::shared_ptr<cs::scene::CelestialObject> satellite =
          std::make_shared<cs::scene::CelestialObject>(sat.bodyId, sat.bodyName);
      satellite->setExistenceAsStrings({sat.existenceStart, sat.existenceEnd});
      satellite->setRadii(glm::dvec3{0.1});
      satellite->setBodyCullingRadius(100.);
      satellite->setOrbitCullingRadius(10000000.);
      satellite->setIsCollidable(false);
      mAllSettings->mObjects.insert(sat.bodyName, satellite);

      Plugin::Settings::Satellite satelliteSettings;
      satelliteSettings.mModelFile      = "../share/resources/models/VLEO_alt.glb";
      satelliteSettings.mEnvironmentMap = "../share/resources/textures/marsEnvMap.dds";
      satelliteSettings.mFieldOfView    = 1.2;
      satelliteSettings.mCameraObject   = sat.bodyName;

      mPluginSettings.mSatellites[sat.bodyName] = satelliteSettings;
      addSatellite(sat.bodyName, mPluginSettings.mSatellites[sat.bodyName]);
    } else {
      // TODO Either recreate the CelestialObject with updated lifetime, or forbid changing it for
      // existing satellites.
    }
    mPendingDownloads.clear();
    if (failed) {
      throw std::runtime_error("Loading satellite kernels failed!");
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::addSatellite(std::string const& name, Settings::Satellite const& settings) {
  mSatellites.push_back(
      std::make_shared<Satellite>(settings, name, mSceneGraph, mAllSettings, mSolarSystem));

  settings.mFieldOfView.connectAndTouch([&](double fov) {
    mGuiManager->getGui()->callJavascript("CosmoScout.satellites.setFieldOfView", name, fov, false);
  });

  mGuiManager->getGui()->callJavascript("CosmoScout.satellites.addSatellite", name,
      mAllSettings->mObjects.at(name)->getCenterName(),
      mAllSettings->mObjects.at(name)->getFrameName());
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::satellites
