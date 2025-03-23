////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "Plugin.hpp"

#include "../../../src/cs-core/GuiManager.hpp"
#include "../../../src/cs-core/SolarSystem.hpp"
#include "../../../src/cs-core/TimeControl.hpp"
#include "../../../src/cs-gui/GuiItem.hpp"
#include "../../../src/cs-utils/logger.hpp"
#include "logger.hpp"

#include <VistaKernel/GraphicsManager/VistaTransformNode.h>
#include <VistaKernelOpenSGExt/VistaOpenSGMaterialTools.h>

////////////////////////////////////////////////////////////////////////////////////////////////////

EXPORT_FN cs::core::PluginBase* create() {
  return new csp::sharad::Plugin;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

EXPORT_FN void destroy(cs::core::PluginBase* pluginBase) {
  delete pluginBase; // NOLINT(cppcoreguidelines-owning-memory)
}

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace csp::sharad {

////////////////////////////////////////////////////////////////////////////////////////////////////

void from_json(nlohmann::json const& j, Plugin::Settings& o) {
  cs::core::Settings::deserialize(j, "anchor", o.mAnchor);
  cs::core::Settings::deserialize(j, "filePath", o.mFilePath);
  cs::core::Settings::deserialize(j, "enabled", o.mEnabled);
}

void to_json(nlohmann::json& j, Plugin::Settings const& o) {
  cs::core::Settings::serialize(j, "anchor", o.mAnchor);
  cs::core::Settings::serialize(j, "filePath", o.mFilePath);
  cs::core::Settings::serialize(j, "enabled", o.mEnabled);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::init() {

  logger().info("Loading plugin...");

  mOnLoadConnection = mAllSettings->onLoad().connect([this]() { onLoad(); });
  mOnSaveConnection = mAllSettings->onSave().connect([this]() { onSave(); });

  mGuiManager->executeJavascriptFile("../share/resources/gui/js/csp-sharad.js");

  mGuiManager->addPluginTabToSideBarFromHTML(
      "SHARAD Profiles", "line_style", "../share/resources/gui/sharad-tab.html");

  mGuiManager->getGui()->registerCallback("sharad.setEnabled",
      "Enables or disables the rendering of SHARAD profiles.",
      std::function([this](bool enable) { mPluginSettings.mEnabled = enable; }));

  mPluginSettings.mFilePath.connect([this](std::string const& filePath) {
    // Delete all old Sharad profiles first.
    for (auto const& node : mSharadNodes) {
      mSceneGraph->GetRoot()->DisconnectChild(node.get());
    }

    mSharads.clear();
    mSharadNodes.clear();

    // Clear UI list.
    mGuiManager->getGui()->callJavascript("CosmoScout.gui.clearHtml", "list-sharad");

    // Then add new ones.
    std::filesystem::path               dir(filePath);
    std::filesystem::directory_iterator end_iter;

    if (std::filesystem::exists(dir) && std::filesystem::is_directory(dir)) {
      for (std::filesystem::directory_iterator dir_iter(dir); dir_iter != end_iter; ++dir_iter) {
        if (std::filesystem::is_regular_file(dir_iter->status())) {
          std::filesystem::path path(std::filesystem::path(*dir_iter).lexically_normal());
          std::string             file(path.stem().string());
          std::string             ext(path.extension().string());

          if (ext == ".tab") {
            std::string sName = file.substr(0, file.length() - 5);
            auto        sharad =
                std::make_shared<Sharad>(mAllSettings, mSolarSystem, mPluginSettings.mAnchor,
                    filePath + sName + "_tiff.tif", filePath + sName + "_geom.tab");

            auto* sharadNode = mSceneGraph->NewOpenGLNode(mSceneGraph->GetRoot(), sharad.get());
            VistaOpenSGMaterialTools::SetSortKeyOnSubtree(
                sharadNode, static_cast<int>(cs::utils::DrawOrder::eOpaqueNonHDR) + 2);
            sharadNode->SetIsEnabled(mPluginSettings.mEnabled.get());

            mSharads.push_back(sharad);
            mSharadNodes.emplace_back(sharadNode);

            mGuiManager->getGui()->callJavascript(
                "CosmoScout.sharad.add", sName, sharad->getStartTime() + 10);
          }
        }
      }
    }
  });

  mPluginSettings.mEnabled.connectAndTouch([this](bool val) {
    for (auto const& node : mSharadNodes) {
      node->SetIsEnabled(val);
    }
  });

  mActiveObjectConnection = mSolarSystem->pActiveObject.connect(
      [this](std::shared_ptr<const cs::scene::CelestialObject> const& body) {
        bool enabled = body == mSolarSystem->getObject(mPluginSettings.mAnchor);

        mGuiManager->getGui()->callJavascript(
            "CosmoScout.sidebar.setTabEnabled", "SHARAD Profiles", enabled);
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

  for (auto const& node : mSharadNodes) {
    mSceneGraph->GetRoot()->DisconnectChild(node.get());
  }

  mGuiManager->removePluginTab("SHARAD Profiles");

  mSolarSystem->pActiveObject.disconnect(mActiveObjectConnection);
  mGuiManager->getGui()->unregisterCallback("sharad.setEnabled");

  mAllSettings->onLoad().disconnect(mOnLoadConnection);
  mAllSettings->onSave().disconnect(mOnSaveConnection);

  logger().info("Unloading done.");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::update() {
  for (auto const& sharad : mSharads) {
    sharad->update(mTimeControl->pSimulationTime.get(), mSolarSystem->getObserver().getScale());
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::onLoad() {
  // Read settings from JSON.
  from_json(mAllSettings->mPlugins.at("csp-sharad"), mPluginSettings);
}
////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::onSave() {
  mAllSettings->mPlugins["csp-sharad"] = mPluginSettings;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::sharad
