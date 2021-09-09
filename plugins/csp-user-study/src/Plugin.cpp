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
#include "../../../src/cs-core/TimeControl.hpp"
#include "../../../src/cs-scene/CelestialAnchorNode.hpp"

#include <VistaKernel/DisplayManager/VistaDisplayManager.h>
#include <VistaKernel/DisplayManager/VistaDisplaySystem.h>
#include <VistaKernel/GraphicsManager/VistaOpenGLNode.h>
#include <VistaKernel/GraphicsManager/VistaSceneGraph.h>
#include <VistaKernel/GraphicsManager/VistaTransformNode.h>
#include <VistaKernel/InteractionManager/VistaUserPlatform.h>
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
  {Plugin::StageType::eCheckpoint, "checkpoint"},
  {Plugin::StageType::eRequestFMS, "requestFMS"},
  {Plugin::StageType::eRequestCOG, "requestCOG"},
  {Plugin::StageType::eMessage, "message"},
  {Plugin::StageType::eSwitchScenario, "switchScenario"},
});

// clang-format on

////////////////////////////////////////////////////////////////////////////////////////////////////

void from_json(nlohmann::json const& j, Plugin::Settings::StageSetting& o) {
  cs::core::Settings::deserialize(j, "type", o.mType);
  cs::core::Settings::deserialize(j, "bookmark", o.mBookmarkName);
  cs::core::Settings::deserialize(j, "scale", o.mScaling);
  cs::core::Settings::deserialize(j, "data", o.mData);
}

void to_json(nlohmann::json& j, Plugin::Settings::StageSetting const& o) {
  cs::core::Settings::serialize(j, "type", o.mType);
  cs::core::Settings::serialize(j, "bookmark", o.mBookmarkName);
  cs::core::Settings::serialize(j, "scale", o.mScaling);
  cs::core::Settings::serialize(j, "data", o.mData);
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
  cs::core::Settings::deserialize(j, "otherScenarios", o.mOtherScenarios);
  cs::core::Settings::deserialize(j, "recordingInterval", o.pRecordingInterval);
  cs::core::Settings::deserialize(j, "stages", o.mStageSettings);
}

void to_json(nlohmann::json& j, Plugin::Settings const& o) {
  cs::core::Settings::serialize(j, "otherScenarios", o.mOtherScenarios);
  cs::core::Settings::serialize(j, "recordingInterval", o.pRecordingInterval);
  cs::core::Settings::serialize(j, "stages", o.mStageSettings);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::init() {

  logger().info("Loading plugin...");

  mOnLoadConnection = mAllSettings->onLoad().connect([this]() { onLoad(); });
  mOnSaveConnection = mAllSettings->onSave().connect(
      [this]() { mAllSettings->mPlugins["csp-user-study"] = *mPluginSettings; });

  mGuiManager->addSettingsSectionToSideBarFromHTML(
      "User Study", "people", "../share/resources/gui/user_study_settings.html");
  mGuiManager->addScriptToGuiFromJS("../share/resources/gui/js/csp-user-study.js");

  mGuiManager->getGui()->registerCallback("userStudy.deleteAllCheckpoints",
      "Deletes all checkpoints and all bookmarks.", std::function([this]() {
        mPluginSettings->mStageSettings.clear();
        while (!mGuiManager->getBookmarks().empty()) {
          mGuiManager->removeBookmark(mGuiManager->getBookmarks().begin()->first);
        }
        updateStages();
      }));

  mGuiManager->getGui()->registerCallback("userStudy.setRecordingInterval",
      "Sets the checkpoint recording interval in seconds.", std::function([this](double val) {
        mPluginSettings->pRecordingInterval = static_cast<uint32_t>(val);
      }));
  mPluginSettings->pRecordingInterval.connectAndTouch(
      [this](uint32_t val) { mGuiManager->setSliderValue("userStudy.setRecordingInterval", val); });

  // Set the mEnableRecording value based on the corresponding checkbox.
  mGuiManager->getGui()->registerCallback("userStudy.setEnableRecording",
      "Enables or disables frame time recording.", std::function([this](bool enable) {
        mEnableRecording = enable;

        if (enable) {
          mGuiManager->getGui()->executeJavascript(
              "document.querySelector('.user-study-record-button').innerHTML = "
              "'<i class=\"material-icons\">stop</i> Stop Recording';");
          mLastRecordTime = std::chrono::steady_clock::now();
        } else {
          mGuiManager->getGui()->executeJavascript(
              "document.querySelector('.user-study-record-button').innerHTML = "
              "'<i class=\"material-icons\">fiber_manual_record</i> Start New Recording';");

          for (std::size_t i = 0; i < mStageViews.size(); i++) {
            setupStage(i);
          }
          updateStages();
        }
      }));

  mGuiManager->getGui()->registerCallback("userStudy.gotoFirst",
      "Teleports to the first checkpoint.", std::function([this]() {
        while (mCurrentStageIdx > 0) {
          previousStage();
        }
        teleportToCurrent();
      }));
  mGuiManager->getGui()->registerCallback("userStudy.gotoPrevious",
      "Teleports to the previous checkpoint.", std::function([this]() {
        previousStage();
        teleportToCurrent();
      }));
  mGuiManager->getGui()->registerCallback("userStudy.gotoNext",
      "Teleports to the next checkpoint.", std::function([this]() {
        nextStage();
        teleportToCurrent();
      }));
  mGuiManager->getGui()->registerCallback("userStudy.gotoLast",
      "Teleports to the last checkpoint.", std::function([this]() {
        while (mCurrentStageIdx < mPluginSettings->mStageSettings.size()-1) {
        nextStage();
        }
        teleportToCurrent();
      }));

  GetVistaSystem()->GetKeyboardSystemControl()->BindAction(VISTA_KEY_BACKSPACE, [this]() {
    if (mInputManager->pSelectedGuiItem.get() && mInputManager->pSelectedGuiItem.get()->getIsKeyboardInputElementFocused()) {
      return;
    }

    teleportToCurrent();
  });

  onLoad();

  logger().info("Loading done.");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::deInit() {
  logger().info("Unloading plugin...");

  // remove stages
  unload();

  mAllSettings->onLoad().disconnect(mOnLoadConnection);
  mAllSettings->onSave().disconnect(mOnSaveConnection);

  mGuiManager->removeSettingsSection("User Study");
  mGuiManager->getGui()->unregisterCallback("userStudy.setRecordingInterval");
  mGuiManager->getGui()->unregisterCallback("userStudy.deleteAllCheckpoints");
  mGuiManager->getGui()->unregisterCallback("userStudy.setEnableRecording");
  mGuiManager->getGui()->unregisterCallback("userStudy.gotoFirst");
  mGuiManager->getGui()->unregisterCallback("userStudy.gotoPrevious");
  mGuiManager->getGui()->unregisterCallback("userStudy.gotoNext");
  mGuiManager->getGui()->unregisterCallback("userStudy.gotoLast");

  logger().info("Unloading done.");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::onLoad() {

  unload();
  mCurrentStageIdx = 0;

  // Read settings from JSON
  from_json(mAllSettings->mPlugins.at("csp-user-study"), *mPluginSettings);

  // Get scenegraph to init stages
  VistaSceneGraph* pSG = GetVistaSystem()->GetGraphicsManager()->GetSceneGraph();
  // Init stages
  for (std::size_t i = 0; i < mStageViews.size(); i++) {
    auto& view = mStageViews[i];
    // Create and register anchor
    view.mAnchor =
        std::make_shared<cs::scene::CelestialAnchorNode>(pSG->GetRoot(), pSG->GetNodeBridge());
    mSolarSystem->registerAnchor(view.mAnchor);

    // Create and setup gui area
    view.mGuiArea = std::make_unique<cs::gui::WorldSpaceGuiArea>(720, 720);
    view.mGuiArea->setUseLinearDepthBuffer(true);

    // Create transform node
    view.mTransformNode =
        std::unique_ptr<VistaTransformNode>(pSG->NewTransformNode(view.mAnchor.get()));

    // Create gui node
    view.mGuiNode = std::unique_ptr<VistaOpenGLNode>(
        pSG->NewOpenGLNode(view.mTransformNode.get(), view.mGuiArea.get()));

    // Register selectable
    mInputManager->registerSelectable(view.mGuiNode.get());

    // Create gui item & attach it to gui area
    view.mGuiItem = std::make_unique<cs::gui::GuiItem>(
        "file://{csp-user-study-cp}../share/resources/gui/user-study-stage.html");
    view.mGuiArea->addItem(view.mGuiItem.get());
    view.mGuiItem->waitForFinishedLoading();
    view.mGuiItem->setZoomFactor(2);

    // register callbacks
    view.mGuiItem->registerCallback("setFMS", "Callback to get slider value",
        std::function([this](double value) { mCurrentFMS = static_cast<uint32_t>(value); }));
    view.mGuiItem->registerCallback(
        "confirmFMS", "Call this to submit the FMS rating", std::function([this]() {
          resultsLogger().info("{}: FMS: {}",
              mPluginSettings->mStageSettings[mCurrentStageIdx].mBookmarkName, mCurrentFMS.get());
          nextStage();
        }));
    view.mGuiItem->registerCallback(
        "confirmMSG", "Call this to advance to the next stage", std::function([this]() {
          resultsLogger().info("{}: MSG", mPluginSettings->mStageSettings[mCurrentStageIdx].mBookmarkName);
          nextStage();
        }));
    view.mGuiItem->registerCallback(
        "loadScenario", "Call this to load a new scenario", std::function([this](std::string path) {
          resultsLogger().info("Loading Scenario at " + path);
          mStageViews[mCurrentStageIdx % mStageViews.size()].mGuiItem->callJavascript("playSound", 0);
          mGuiManager->getGui()->callJavascript("CosmoScout.callbacks.core.load", path);
        }));
    view.mGuiItem->registerCallback("setEnableCOGMeasurement",
        "Enables or disables center of gravity recording.", std::function([this](bool enable) {
          mEnableCOGMeasurement = enable;

          if (!enable) {
            nextStage();
          }
        }));

    setupStage(i);
  }

  // register FMS callback part afterwards
  mCurrentFMS.connectAndTouch([this](uint32_t value) {
    for (std::size_t i = 0; i < mStageViews.size(); i++) {
      mStageViews[i].mGuiItem->callJavascript("CosmoScout.gui.setSliderValue", "setFMS", false, value);
    }
  });

  updateStages();

  // Mark start of scenario in results log
  resultsLogger().info("Scenario started");
}


////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::unload() {
  VistaSceneGraph* pSG = GetVistaSystem()->GetGraphicsManager()->GetSceneGraph();
  for (auto& view : mStageViews) {
    // skip unload if view is empty
    if (view.mAnchor == nullptr) {
      break;
    }
    // unregister callbacks
    view.mGuiItem->unregisterCallback("setFMS");
    view.mGuiItem->unregisterCallback("confirmFMS");
    view.mGuiItem->unregisterCallback("confirmMSG");
    view.mGuiItem->unregisterCallback("loadScenario");
    view.mGuiItem->unregisterCallback("setEnableCOGMeasurement");
    // disconnect from scene graph
    pSG->GetRoot()->DisconnectChild(view.mAnchor.get());
    view.mAnchor->DisconnectChild(view.mTransformNode.get());
    view.mTransformNode->DisconnectChild(view.mGuiNode.get());
    // unregister anchor
    mSolarSystem->unregisterAnchor(view.mAnchor);
    // unregister selectable
    mInputManager->unregisterSelectable(view.mGuiNode.get());

    view = {};
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::update() {
  if (mEnableRecording) {
    auto now = std::chrono::steady_clock::now();
    if (std::chrono::duration_cast<std::chrono::seconds>(now - mLastRecordTime).count() >=
        mPluginSettings->pRecordingInterval.get()) {
      mLastRecordTime = now;

      if (mPluginSettings->mStageSettings.size() % 10 == 0) {
        mStageViews[0].mGuiItem->callJavascript("playSound", 2);
      } else {
        mStageViews[0].mGuiItem->callJavascript("playSound", 0);
      }

      cs::core::Settings::Bookmark bookmark;
      bookmark.mName =
          "user-study-bookmark-" + std::to_string(mPluginSettings->mStageSettings.size());
      bookmark.mLocation = {this->mSolarSystem->getObserver().getCenterName(),
          this->mSolarSystem->getObserver().getFrameName(),
          this->mSolarSystem->getObserver().getAnchorPosition(),
          this->mSolarSystem->getObserver().getAnchorRotation()};

      mGuiManager->addBookmark(bookmark);

      Settings::StageSetting stage;
      stage.mScaling      = static_cast<float>(this->mSolarSystem->getObserver().getAnchorScale());
      stage.mBookmarkName = bookmark.mName;
      mPluginSettings->mStageSettings.push_back(stage);

      logger().info("Recorded Checkpoint {}.", bookmark.mName);
    }

  } else {

    if (mEnableCOGMeasurement) {
      auto const& name = mPluginSettings->mStageSettings[mCurrentStageIdx].mBookmarkName;

      auto* platform = GetVistaSystem()
              ->GetPlatformFor(GetVistaSystem()->GetDisplayManager()->GetDisplaySystem())
              ->GetPlatformNode();
      auto* pProps = GetVistaSystem()->GetDisplayManager()->GetDisplaySystem()->GetDisplaySystemProperties();
      auto translation = platform->GetTranslation();
       auto rotation = pProps->GetViewerOrientation().GetAngles();
      resultsLogger().trace("{}: COG {} {} {} {} {} {}", name, -translation[0], -translation[1], -translation[2],  rotation.a, rotation.b, rotation.c);
    }

    // check if current stage is normal checkpoint
    if (mPluginSettings->mStageSettings[mCurrentStageIdx].mType == Plugin::StageType::eCheckpoint) {
      // check distance to CP
      glm::dvec3 vecToObserver = mStageViews[mCurrentStageIdx % mStageViews.size()].mAnchor->getRelativePosition(
          mTimeControl->pSimulationTime.get(), mSolarSystem->getObserver());
      if (glm::length(vecToObserver) < 1.0) {
        // go to next stage
        resultsLogger().info("{}: Passed Checkpoint",
            mPluginSettings->mStageSettings[mCurrentStageIdx].mBookmarkName);
        nextStage();

          mStageViews[mCurrentStageIdx % mStageViews.size()].mGuiItem->callJavascript("playSound", 0);
      }
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::setupStage(std::size_t stageIdx) {
  auto const& settings = mPluginSettings->mStageSettings[stageIdx];

  // Fetch stage at Index
  auto& view = mStageViews[(stageIdx) % mStageViews.size()];
  // Fetch bookmark for position
  std::optional<cs::core::Settings::Bookmark> bookmark =
      getBookmarkByName(settings.mBookmarkName);
  if (bookmark.has_value()) {
    cs::core::Settings::Bookmark b = bookmark.value();
    // Move anchor to bookmark position
    view.mAnchor->setCenterName(b.mLocation->mCenter);
    view.mAnchor->setFrameName(b.mLocation->mFrame);
    view.mAnchor->setAnchorScale(settings.mScaling);
    if (b.mLocation->mPosition.has_value()) {
      view.mAnchor->setAnchorPosition(b.mLocation->mPosition.value());
    }
    if (b.mLocation->mRotation.has_value()) {
      view.mAnchor->setAnchorRotation(b.mLocation->mRotation.value());
    }

    // Add Scaling factor
    const float checkPointScale = 2.f;
    view.mTransformNode->SetScale(
         checkPointScale,  checkPointScale, 1.0F);

    // Set webview according to type
    switch (settings.mType) {
    case StageType::eCheckpoint: {
      view.mGuiItem->callJavascript("reset");
      break;
    }
    case StageType::eRequestFMS: {
      view.mGuiItem->callJavascript("setFMS");
      break;
    }
    case StageType::eRequestCOG: {
      view.mGuiItem->callJavascript("setCOG");
      break;
    }
    case StageType::eMessage: {
      view.mGuiItem->callJavascript("setMSG", settings.mData.value_or(""));
      break;
    }
    case StageType::eSwitchScenario: {
      std::string html = "";
      for (Plugin::Settings::Scenario& scenario : mPluginSettings->mOtherScenarios) {
        html += "<input class=\"btn\" type=\"button\" value=\"" + scenario.mName +
                "\" onclick=\"window.callNative('loadScenario', '" + scenario.mPath +
                "')\">\n";
      }
      view.mGuiItem->callJavascript("setCHS", html);
      break;
    }
    }
  }
}


////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::teleportToCurrent() {
  auto const& settings = mPluginSettings->mStageSettings[std::max(0, static_cast<int>(mCurrentStageIdx)-1)];
  auto bookmark = getBookmarkByName(settings.mBookmarkName);
  
  if (bookmark.has_value()) {
    cs::core::Settings::Bookmark b = bookmark.value();

    if (b.mLocation) {
      auto loc = b.mLocation.value();

      if (loc.mRotation.has_value() && loc.mPosition.has_value()) {
        mSolarSystem->flyObserverTo(loc.mCenter, loc.mFrame, loc.mPosition.value(),
            loc.mRotation.value(), 0.0);
      } else if (loc.mPosition.has_value()) {
        mSolarSystem->flyObserverTo(
            loc.mCenter, loc.mFrame, loc.mPosition.value(), 0.0);
      } else {
        mSolarSystem->flyObserverTo(loc.mCenter, loc.mFrame, 0.0);
      }
    }
  }
}


////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::nextStage() {

  // Advance the current stage index.
  mCurrentStageIdx = std::min(mCurrentStageIdx + 1, mPluginSettings->mStageSettings.size() - 1);

  logger().info("Reached stage {}", mCurrentStageIdx);

  // Setup the stage which becomes visible next.
  std::size_t newlyVisibleIdx = mCurrentStageIdx + mStageViews.size() - 1;
  if (newlyVisibleIdx < mPluginSettings->mStageSettings.size()) {
    setupStage(newlyVisibleIdx);
  }

  updateStages();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::previousStage() {

  // Advance the current stage index.
  mCurrentStageIdx = std::max(static_cast<int>(mCurrentStageIdx) - 1, 0);

  logger().info("Reached stage {}", mCurrentStageIdx);

  // Setup the stage which becomes visible next.
  std::size_t newlyVisibleIdx = mCurrentStageIdx;
  if (newlyVisibleIdx < mPluginSettings->mStageSettings.size()) {
    setupStage(newlyVisibleIdx);
  }

  updateStages();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::updateStages() {

  // Set css classes of all visible stages. The item at i==0 is the current stage, and
  // i==mStageViews.size()-1 is the most distant stage.
  for (std::size_t i = 0; i < mStageViews.size(); i++) {
    auto stageIdx = (mCurrentStageIdx + i) % mStageViews.size();
    if (mCurrentStageIdx + i < mPluginSettings->mStageSettings.size()) {
      mStageViews[stageIdx].mGuiItem->callJavascript("setBodyClass", "stage" + std::to_string(i));
    } else {
      mStageViews[stageIdx].mGuiItem->callJavascript("setBodyClass", "hidden");
    }

    // Make only current stage interactive.
    mStageViews[stageIdx].mGuiItem->setIsInteractive(i == 0);

    // Ensure that the checkpoints are drawn back-to-front.
    std::size_t sortKey =
        static_cast<std::size_t>(cs::utils::DrawOrder::eTransparentItems) + mStageViews.size() - i;
    VistaOpenSGMaterialTools::SetSortKeyOnSubtree(
        mStageViews[stageIdx].mGuiNode.get(), static_cast<int>(sortKey));
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

} // namespace csp::userstudy
