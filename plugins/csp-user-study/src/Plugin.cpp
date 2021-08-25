////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "Plugin.hpp"
#include "logger.hpp"
#include "resultsLogger.hpp"

#include "../../../src/cs-core/GuiManager.hpp"
#include "../../../src/cs-core/InputManager.hpp"
#include "../../../src/cs-core/SolarSystem.hpp"
#include "../../../src/cs-scene/CelestialAnchorNode.hpp"

#include <VistaKernel/GraphicsManager/VistaOpenGLNode.h>
#include <VistaKernel/GraphicsManager/VistaSceneGraph.h>
#include <VistaKernel/VistaSystem.h>
#include <VistaKernelOpenSGExt/VistaOpenSGMaterialTools.h>

#include <optional>

////////////////////////////////////////////////////////////////////////////////////////////////////

EXPORT_FN cs::core::PluginBase* create() {
  return new csp::userstudy::Plugin;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

EXPORT_FN void destroy(cs::core::PluginBase* pluginBase) {
  delete pluginBase; // NOLINT(cppcoreguidelines-owning-memory)
}

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace csp::userstudy {

////////////////////////////////////////////////////////////////////////////////////////////////////

// clang-format off

// NOLINTNEXTLINE
NLOHMANN_JSON_SERIALIZE_ENUM(Plugin::StageType, {
  {Plugin::StageType::eNone, nullptr},
  {Plugin::StageType::eCheckpoint, "checkpoint"},
  {Plugin::StageType::eRequestFMS, "requestFMS"},
  {Plugin::StageType::eSwitchScenario, "switchScenario"},
});

// clang-format on

////////////////////////////////////////////////////////////////////////////////////////////////////

void from_json(nlohmann::json const& j, Plugin::Settings::StageSetting& o) {
  cs::core::Settings::deserialize(j, "type", o.mType);
  cs::core::Settings::deserialize(j, "bookmark", o.mBookmarkName);
  cs::core::Settings::deserialize(j, "scale", o.mScaling);

  if (o.mType.get() == Plugin::StageType::eNone) {
    throw cs::core::Settings::DeserializationException(
        "'type'", "Invalid stage type given! Should be one of the types outlined in the README.md");
  }
}

void to_json(nlohmann::json& j, Plugin::Settings::StageSetting const& o) {
  cs::core::Settings::serialize(j, "type", o.mType);
  cs::core::Settings::serialize(j, "bookmark", o.mBookmarkName);
  cs::core::Settings::serialize(j, "scale", o.mScaling);
}

void from_json(nlohmann::json const& j, Plugin::Settings::Scenario& o) {
  cs::core::Settings::deserialize(j, "name", o.mName);
  cs::core::Settings::deserialize(j, "path", o.mPath);
}

void to_json(nlohmann::json& j, Plugin::Settings::Scenario const& o) {
  cs::core::Settings::serialize(j, "name", o.mName);
  cs::core::Settings::serialize(j, "path", o.mPath);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void from_json(nlohmann::json const& j, Plugin::Settings& o) {
  cs::core::Settings::deserialize(j, "enabled", o.mEnabled);
  cs::core::Settings::deserialize(j, "debug", o.mDebug);
  cs::core::Settings::deserialize(j, "otherScenarios", o.mOtherScenarios);
  cs::core::Settings::deserialize(j, "stages", o.mStageSettings);
}

void to_json(nlohmann::json& j, Plugin::Settings const& o) {
  cs::core::Settings::serialize(j, "enabled", o.mEnabled);
  cs::core::Settings::serialize(j, "debug", o.mDebug);
  cs::core::Settings::serialize(j, "otherScenarios", o.mOtherScenarios);
  cs::core::Settings::serialize(j, "stages", o.mStageSettings);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool Plugin::Settings::Scenario::operator==(Plugin::Settings::Scenario const& other) const {
  return mName.get() == other.mName.get() && mPath.get() == other.mPath.get();
}

bool Plugin::Settings::StageSetting::operator==(Plugin::Settings::StageSetting const& other) const {
  return mType.get() == other.mType.get() && mBookmarkName.get() == other.mBookmarkName.get() &&
         mScaling.get() == other.mScaling.get();
}

bool Plugin::Settings::operator==(Plugin::Settings const& other) const {
  return mOtherScenarios == other.mOtherScenarios && mStageSettings == other.mStageSettings;
}

bool Plugin::Settings::operator!=(Plugin::Settings const& other) const {
  return !(*this == other);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::init() {

  logger().info("Loading plugin...");

  mOnLoadConnection = mAllSettings->onLoad().connect([this]() { onLoad(); });
  mOnSaveConnection = mAllSettings->onSave().connect(
      [this]() { mAllSettings->mPlugins["csp-user-study"] = *mPluginSettings; });

  mGuiManager->addScriptToGuiFromJS("../share/resources/gui/js/csp-user-study.js");

  onLoad();

  logger().info("Loading done.");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::update() {
  if (mPluginSettings->mEnabled.get()) {
    // TODO: check active stage index & which stages to hide/un-hide
    // TODO: check flythrough for checkpoints?
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::deInit() {
  logger().info("Unloading plugin...");

  // remove stages
  unload(*mPluginSettings);

  mAllSettings->onLoad().disconnect(mOnLoadConnection);
  mAllSettings->onSave().disconnect(mOnSaveConnection);

  logger().info("Unloading done.");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::onLoad() {
  // Read settings from JSON
  from_json(mAllSettings->mPlugins.at("csp-user-study"), *mPluginSettings);

  // Get scenegraph to init stages
  VistaSceneGraph* pSG = GetVistaSystem()->GetGraphicsManager()->GetSceneGraph();
  // Init stages
  for (int i = 0; i <= 1; i++)
  {
    mStages.emplace_back();
    // Create and register anchor
    mStages[i].mAnchor = std::make_shared<cs::scene::CelestialAnchorNode>(pSG->GetRoot(), pSG->GetNodeBridge());
    mSolarSystem->registerAnchor(mStages[i].mAnchor);
    // Create and setup gui area
    mStages[i].mGuiArea = std::make_unique<cs::gui::WorldSpaceGuiArea>(720, 720);
    mStages[i].mGuiArea->setUseLinearDepthBuffer(true);
    // Create transform node
    mStages[i].mTransform = std::unique_ptr<VistaTransformNode>(pSG->NewTransformNode(mStages[i].mAnchor.get()));
    // Create gui node
    mStages[i].mGuiNode = std::unique_ptr<VistaOpenGLNode>(pSG->NewOpenGLNode(mStages[i].mTransform.get(), mStages[i].mGuiArea.get()));
    // Register selectable
    mInputManager->registerSelectable(mStages[i].mGuiNode.get());
    // Set sort key
    VistaOpenSGMaterialTools::SetSortKeyOnSubtree(mStages[i].mGuiNode.get(), static_cast<int>(cs::utils::DrawOrder::eTransparentItems));
    // Create gui item & attach it to gui area
    mStages[i].mGuiItem = std::make_unique<cs::gui::GuiItem>("file://../share/resources/gui/user-study-stage.html");
    mStages[i].mGuiArea->addItem(mStages[i].mGuiItem.get());
    mStages[i].mGuiItem->waitForFinishedLoading();
  }
  
  // Setup first two checkpoints
  setupStage(mPluginSettings->mStageSettings[0], 0, true);
  setupStage(mPluginSettings->mStageSettings[1], 1);

  resultsLogger().info("Scenario started");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::setupStage(Plugin::Settings::StageSetting settings, uint32_t atStagesIdx, bool isCurrent) {
  // Fetch stage at Index
  Stage &stage = mStages[atStagesIdx];
  // Fetch bookmark for position
  std::optional<cs::core::Settings::Bookmark> bookmark = getBookmarkByName(settings.mBookmarkName.get());
  if (bookmark.has_value())
  {
    cs::core::Settings::Bookmark b = bookmark.value();
    // Move anchor to bookmark position
    stage.mAnchor->setCenterName(b.mLocation->mCenter);
    stage.mAnchor->setFrameName(b.mLocation->mFrame);
    if (b.mLocation->mPosition.has_value())
    {
      stage.mAnchor->setAnchorPosition(b.mLocation->mPosition.value());
    }
    if (b.mLocation->mPosition.has_value())
    {
      stage.mAnchor->setAnchorRotation(b.mLocation->mRotation.value());
    }

    // Add Scaling factor
    stage.mTransform->Scale(settings.mScaling.get() * static_cast<float>(stage.mGuiArea->getWidth()),
      settings.mScaling.get() * static_cast<float>(stage.mGuiArea->getHeight()), 1.0F);

    // Set webview according to type
    switch (settings.mType.get())
    {
    case StageType::eCheckpoint :
      stage.mGuiItem->callJavascript("setCP");
      break;
    
    case StageType::eRequestFMS :
      stage.mGuiItem->callJavascript("setFMS");
      break;

    case StageType::eSwitchScenario :
      stage.mGuiItem->callJavascript("setCHS");
      break;

    default:
      logger().error("Unrecognized stage type when setting up Stage! stage type: {}", settings.mType.get());
      break;
    }

    // Set opacity to 1.0 if isCurrent, else set to 0.5
    stage.mGuiItem->callJavascript("setOpacity", isCurrent ? 1.0 : 0.5);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::unload(Plugin::Settings pluginSettings) {
  VistaSceneGraph* pSG = GetVistaSystem()->GetGraphicsManager()->GetSceneGraph();
  for (Stage &stage : mStages) {
    // unload and disconnect each stage
    //stage.unload(mSolarSystem, mInputManager);
  }
  // clear stages list
  //mStages.clear();
}

/////////////////////////////////////////////////////////////////////////////////////////////////////

std::optional<cs::core::Settings::Bookmark> Plugin::getBookmarkByName(std::string name) {
  cs::core::Settings::Bookmark bookmark;
  for (auto it = mGuiManager->getBookmarks().begin(); it != mGuiManager->getBookmarks().end();
      ++it) {
    if (it->second.mName == name) {
      return it->second;
    }
  }
  logger().error("No bookmark with the name \"" + name + "\" could be found!");
  return std::nullopt;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::userstudy
