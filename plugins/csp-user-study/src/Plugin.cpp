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
#include "../../../src/cs-core/TimeControl.hpp"

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

bool Plugin::Settings::StageSetting::operator!=(Plugin::Settings::StageSetting const& other) const {
  return !(*this == other);
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

  // connect onBookmarkAdded 
  mOnBookmarkAddedConnection = mGuiManager->onBookmarkAdded().connect([this](uint32_t, cs::core::Settings::Bookmark const&) { logger().info(this->mSolarSystem->getObserver().getAnchorScale()); });

  onLoad();

  logger().info("Loading done.");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::update() {
  if (mPluginSettings->mEnabled.get()) {
    // check if current stage is normal checkpoint
    if (mPluginSettings->mStageSettings[mStageIdx].mType.get() == Plugin::StageType::eCheckpoint)
    {
      // check distance to CP
      glm::dvec3 vecToObserver = mStages[mStageIdx].mAnchor->getRelativePosition(mTimeControl->pSimulationTime.get() , mSolarSystem->getObserver());
      if (glm::length(vecToObserver) < mPluginSettings->mStageSettings[mStageIdx].mScaling.get())
      {
        logger().trace("Observer within CP range");
        // go to next stage
        advanceStage();
      }
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::deInit() {
  logger().info("Unloading plugin...");

  // remove stages
  unload();

  mAllSettings->onLoad().disconnect(mOnLoadConnection);
  mAllSettings->onSave().disconnect(mOnSaveConnection);

  // disconnect onBookmarkAdded
  mGuiManager->onBookmarkAdded().disconnect(mOnBookmarkAddedConnection);

  logger().info("Unloading done.");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::onLoad() {
  logger().trace("User Study onLoad");

  unload();
  logger().trace(__LINE__);
  mStageIdx = 0;

  // Read settings from JSON
  from_json(mAllSettings->mPlugins.at("csp-user-study"), *mPluginSettings);

  // Get scenegraph to init stages
  VistaSceneGraph* pSG = GetVistaSystem()->GetGraphicsManager()->GetSceneGraph();
  // Init stages
  for (int i = 0; i < mStages.size(); i++)
  {
    logger().trace("inside array init {}", i);
    Stage &stage = mStages[i];
    // Create and register anchor
    stage.mAnchor = std::make_shared<cs::scene::CelestialAnchorNode>(pSG->GetRoot(), pSG->GetNodeBridge());
    mSolarSystem->registerAnchor(mStages[i].mAnchor);
    logger().trace(__LINE__);
    // Create and setup gui area
    stage.mGuiArea = std::make_unique<cs::gui::WorldSpaceGuiArea>(720, 720);
    stage.mGuiArea->setUseLinearDepthBuffer(true);
    logger().trace(__LINE__); //last heartbeat
    // Create transform node
    stage.mTransform = std::unique_ptr<VistaTransformNode>(pSG->NewTransformNode(mStages[i].mAnchor.get()));
    logger().trace(__LINE__);
    // Create gui node
    stage.mGuiNode = std::unique_ptr<VistaOpenGLNode>(pSG->NewOpenGLNode(mStages[i].mTransform.get(), mStages[i].mGuiArea.get()));
    logger().trace(__LINE__);
    // Register selectable
    mInputManager->registerSelectable(mStages[i].mGuiNode.get());
    logger().trace(__LINE__);
    // Set sort key
    VistaOpenSGMaterialTools::SetSortKeyOnSubtree(mStages[i].mGuiNode.get(), static_cast<int>(cs::utils::DrawOrder::eTransparentItems));
    logger().trace(__LINE__);
    // Create gui item & attach it to gui area
    stage.mGuiItem = std::make_unique<cs::gui::GuiItem>("file://{csp-user-study-cp}../share/resources/gui/user-study-stage.html");
    stage.mGuiArea->addItem(mStages[i].mGuiItem.get());
    stage.mGuiItem->waitForFinishedLoading();

    stage.mGuiItem->setZoomFactor(2);
    logger().trace(__LINE__);
    // register callbacks
    stage.mGuiItem->registerCallback("setFMS", "Callback to get slider value",
      std::function([this](double value) {
        mCurrentFMS = static_cast<uint32_t>(value);
      })
    );
    stage.mGuiItem->registerCallback("confirmFMS", "Call this to submit the FMS rating",
      std::function([this](){
        resultsLogger().info("FMS score submitted: {}", mCurrentFMS.get());
        advanceStage();
      })
    );
    stage.mGuiItem->registerCallback("loadScenario", "Call this to load a new scenario",
      std::function([this](std::string path) {
        resultsLogger().info("Loading Scenario at " + path);
        mGuiManager->getGui()->callJavascript("CosmoScout.callbacks.core.load", path);
      })
    );
  }
  logger().trace(__LINE__);
  // register FMS callback part afterwards
  // TODO: for-loop over mStages.size()
  mCurrentFMS.connectAndTouch([this](uint32_t value) {
      mStages[0].mGuiItem->callJavascript("CosmoScout.gui.setSliderValue", "setFMS", false, value);
      mStages[1].mGuiItem->callJavascript("CosmoScout.gui.setSliderValue", "setFMS", false, value);
    });

  logger().trace(__LINE__);
  // Setup first two checkpoints
  setupStage(0);
  setupStage(1);

  // Mark start of scenario in results log
  resultsLogger().info("Scenario started");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::setupStage(uint32_t stageIdx) {
  logger().trace("Setting up Stage at Idx: {}, using nodes from {}", stageIdx, (stageIdx)%mStages.size());
  auto const& settings = mPluginSettings->mStageSettings[stageIdx];
 
  // Fetch stage at Index
  Stage &stage = mStages[(stageIdx)%mStages.size()];
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
    stage.mTransform->Scale(settings.mScaling.get(), settings.mScaling.get(), 1.0F);

    // Set webview according to type
    switch (settings.mType.get())
      {
      case StageType::eCheckpoint :
      {
        stage.mGuiItem->callJavascript("setCP");
        break;
      }
      case StageType::eRequestFMS :
      {
        stage.mGuiItem->callJavascript("setFMS");
        break;
      }
      case StageType::eSwitchScenario :
      {
        std::string html = "";
        for (Plugin::Settings::Scenario &scenario : mPluginSettings->mOtherScenarios) {
          html += "<input type=\"button\" value=\"" + scenario.mName.get() +  "\" onclick=\"window.callNative('loadScenario', '" + scenario.mPath.get() + "')\">\n";
        }
        stage.mGuiItem->callJavascript("setCHS", html);
        break;
      }
      default:
      {
        logger().error("Unrecognized stage type when setting up Stage! stage type: {}", settings.mType.get());
        break;
      }
    }

    // Set opacity to 1.0 if isCurrent, else set to 0.5
     bool isCurrent = stageIdx == mStageIdx;
    stage.mGuiItem->callJavascript("setOpacity", isCurrent ? 1.0 : 0.5);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::unload() {
  logger().trace(__LINE__);
  VistaSceneGraph* pSG = GetVistaSystem()->GetGraphicsManager()->GetSceneGraph();
  logger().trace(__LINE__);
  for (Stage &stage : mStages) {
    logger().trace("inside unload loop");
    // skip unload if Stage is empty
    if (stage.mAnchor == nullptr)
    {
      logger().trace("skipped unload");
      break;
    }
    // unregister callbacks
    stage.mGuiItem->unregisterCallback("setFMS");
    stage.mGuiItem->unregisterCallback("confirmFMS");
    stage.mGuiItem->unregisterCallback("loadScenario");
    logger().trace("after unregister callbacks");
    // disconnect from scene graph
    pSG->GetRoot()->DisconnectChild(stage.mAnchor.get());
    logger().trace("after disconnect scene graph");
    // unregister anchor
    mSolarSystem->unregisterAnchor(stage.mAnchor);
    logger().trace("after unregister anchor");
    // unregister selectable
    mInputManager->unregisterSelectable(stage.mGuiNode.get());
    logger().trace("end of unload loop / after unregister selectable");
    stage.mGuiArea.reset(nullptr);
    logger().trace(__LINE__);

    stage.mGuiItem.reset(nullptr);
    logger().trace(__LINE__);

    stage.mAnchor.reset();
    logger().trace(__LINE__); //LH
    
    stage.mTransform.reset(nullptr);
    logger().trace(__LINE__);
    
    stage.mGuiNode.reset(nullptr);
    logger().trace(__LINE__);
    
  }

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

void Plugin::advanceStage() {
  if (mStageIdx < mPluginSettings->mStageSettings.size()-1)
  {
    ++mStageIdx;
  }
  // setup next stage if current not last stage
  if (mStageIdx < mPluginSettings->mStageSettings.size()-1) {
    // setup following stage
    setupStage(mStageIdx+1);
    logger().trace("New stage added at Idx: {}", (mStageIdx+1)%mStages.size());
  }
  else {
    // if current is last stage hide other stage
    mStages[(mStageIdx+1)%mStages.size()].mGuiItem->callJavascript("setOpacity", 0.0);
  }
  mStages[(mStageIdx)%mStages.size()].mGuiItem->callJavascript("setOpacity", 1.0);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::userstudy
