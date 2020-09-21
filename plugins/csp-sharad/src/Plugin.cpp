////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "Plugin.hpp"

#include "../../../src/cs-core/GuiManager.hpp"
#include "../../../src/cs-core/SolarSystem.hpp"
#include "../../../src/cs-gui/GuiItem.hpp"
#include "../../../src/cs-utils/logger.hpp"
#include "logger.hpp"

#include <VistaKernel/GraphicsManager/VistaTransformNode.h>
#include <VistaKernelOpenSGExt/VistaOpenSGMaterialTools.h>
#include <boost/filesystem.hpp>

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

  mOnSaveConnection = mAllSettings->onSave().connect(
      [this]() { mAllSettings->mPlugins["csp-sharad"] = mPluginSettings; });

  mGuiManager->addHtmlToGui("sharad", "../share/resources/gui/sharad-template.html");

  mGuiManager->addScriptToGuiFromJS("../share/resources/gui/js/csp-sharad.js");

  mGuiManager->addPluginTabToSideBarFromHTML(
      "SHARAD Profiles", "line_style", "../share/resources/gui/sharad-tab.html");

  mGuiManager->getGui()->registerCallback("sharad.setEnabled",
      "Enables or disables the rendering of SHARAD profiles.",
      std::function([this](bool enable) { mPluginSettings.mEnabled = enable; }));

  mPluginSettings.mFilePath.connect([this](std::string const& filePath) {
    // Delete all old Sharad profiles first.
    for (auto const& sharad : mSharads) {
      mSolarSystem->unregisterAnchor(sharad);
    }

    for (auto const& node : mSharadNodes) {
      mSceneGraph->GetRoot()->DisconnectChild(node.get());
    }

    mSharads.clear();
    mSharadNodes.clear();

    // Clear UI list.
    mGuiManager->getGui()->callJavascript("CosmoScout.gui.clearHtml", "list-sharad");

    // Then add new ones.
    boost::filesystem::path               dir(filePath);
    boost::filesystem::directory_iterator end_iter;

    if (boost::filesystem::exists(dir) && boost::filesystem::is_directory(dir)) {
      for (boost::filesystem::directory_iterator dir_iter(dir); dir_iter != end_iter; ++dir_iter) {
        if (boost::filesystem::is_regular_file(dir_iter->status())) {
          boost::filesystem::path path(boost::filesystem::path(*dir_iter).normalize());
          std::string             file(path.stem().string());
          std::string             ext(path.extension().string());

          if (ext == ".tab") {
            std::string sName  = file.substr(0, file.length() - 5);
            auto        sharad = std::make_shared<Sharad>(mAllSettings, mPluginSettings.mAnchor,
                filePath + sName + "_tiff.tif", filePath + sName + "_geom.tab");
            mSolarSystem->registerAnchor(sharad);

            auto* sharadNode = mSceneGraph->NewOpenGLNode(mSceneGraph->GetRoot(), sharad.get());
            VistaOpenSGMaterialTools::SetSortKeyOnSubtree(
                sharadNode, static_cast<int>(cs::utils::DrawOrder::eOpaqueNonHDR) + 2);
            sharadNode->SetIsEnabled(mPluginSettings.mEnabled.get());

            mSharads.push_back(sharad);
            mSharadNodes.emplace_back(sharadNode);

            mGuiManager->getGui()->callJavascript(
                "CosmoScout.sharad.add", sName, sharad->getExistence()[0] + 10);
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

  mActiveBodyConnection = mSolarSystem->pActiveBody.connectAndTouch(
      [this](std::shared_ptr<cs::scene::CelestialBody> const& body) {
        bool enabled = false;

        if (body && body->getCenterName() == "Mars") {
          enabled = true;
        }

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

  for (auto const& sharad : mSharads) {
    mSolarSystem->unregisterAnchor(sharad);
  }

  for (auto const& node : mSharadNodes) {
    mSceneGraph->GetRoot()->DisconnectChild(node.get());
  }

  mGuiManager->removePluginTab("SHARAD Profiles");

  mSolarSystem->pActiveBody.disconnect(mActiveBodyConnection);
  mGuiManager->getGui()->unregisterCallback("sharad.setEnabled");
  mGuiManager->getGui()->callJavascript("CosmoScout.gui.unregisterHtml", "sharad");

  mAllSettings->onLoad().disconnect(mOnLoadConnection);
  mAllSettings->onSave().disconnect(mOnSaveConnection);

  logger().info("Unloading done.");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::onLoad() {
  // Read settings from JSON.
  from_json(mAllSettings->mPlugins.at("csp-sharad"), mPluginSettings);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::sharad
