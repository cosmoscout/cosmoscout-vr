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
}

void to_json(nlohmann::json& j, Plugin::Settings::Satellite const& o) {
  cs::core::Settings::serialize(j, "modelFile", o.mModelFile);
  cs::core::Settings::serialize(j, "environmentMap", o.mEnvironmentMap);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void from_json(nlohmann::json const& j, Plugin::Settings& o) {
  cs::core::Settings::deserialize(j, "satellites", o.mSatellites);
}

void to_json(nlohmann::json& j, Plugin::Settings const& o) {
  cs::core::Settings::serialize(j, "satellites", o.mSatellites);
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
      std::function([this](std::string jobId) { downloadSatelliteKernel(jobId); }));

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
  if (!mPendingKernels.empty() && mDownloader.hasFinished()) {
    loadSatelliteKernel();
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::onLoad() {

  mSatellites.clear();

  // Read settings from JSON.
  mPluginSettings = mAllSettings->mPlugins.at("csp-satellites");

  for (auto const& settings : mPluginSettings.mSatellites) {
    mSatellites.push_back(std::make_shared<Satellite>(
        settings.second, settings.first, mSceneGraph, mAllSettings, mSolarSystem));
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::onSave() {
  mAllSettings->mPlugins["csp-satellites"] = mPluginSettings;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::downloadSatelliteKernel(std::string const& jobId) {
  std::stringstream downloadPath;
  downloadPath << "http://localhost:8000/jobs/" << jobId << "/result/bsp";
  std::stringstream localPath;
  localPath << "./spice_out/" << jobId << ".bsp";
  mDownloader.download(downloadPath.str(), localPath.str());
  mPendingKernels.push_back(localPath.str());
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::loadSatelliteKernel() {
  bool failed = false;
  for (std::string const& kernel : mPendingKernels) {
    // Load the spice kernels.
    furnsh_c(kernel.c_str());

    if (failed_c()) {
      int32_t const maxSpiceErrorLength = 320;

      std::array<SpiceChar, maxSpiceErrorLength> msg{};
      getmsg_c("LONG", maxSpiceErrorLength, msg.data());
      logger().error(msg.data());
      failed = true;
    }
  }
  mPendingKernels.clear();
  if (failed) {
    throw std::runtime_error("Loading satellite kernels failed!");
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::satellites
