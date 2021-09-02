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
  {Plugin::StageType::eSwitchScenario, "switchScenario"},
});

// clang-format on

////////////////////////////////////////////////////////////////////////////////////////////////////

void from_json(nlohmann::json const& j, Plugin::Settings::StageSetting& o) {
  cs::core::Settings::deserialize(j, "type", o.mType);
  cs::core::Settings::deserialize(j, "bookmark", o.mBookmarkName);
  cs::core::Settings::deserialize(j, "scale", o.mScaling);
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

          for (std::size_t i = 0; i < mStages.size(); i++) {
            setupStage(i);
          }
          updateStages();
        }
      }));

  onLoad();

  logger().info("Loading done.");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::update() {
  if (mEnableRecording) {
    auto now = std::chrono::steady_clock::now();
    if (std::chrono::duration_cast<std::chrono::seconds>(now - mLastRecordTime).count() >=
        mPluginSettings->pRecordingInterval.get()) {
      mLastRecordTime = now;

      cs::core::Settings::Bookmark bookmark;
      bookmark.mName =
          "user-study-bookmark-" + std::to_string(mPluginSettings->mStageSettings.size());
      bookmark.mLocation = {this->mSolarSystem->getObserver().getCenterName(),
          this->mSolarSystem->getObserver().getFrameName(),
          this->mSolarSystem->getObserver().getAnchorPosition(),
          this->mSolarSystem->getObserver().getAnchorRotation()};

      mGuiManager->addBookmark(bookmark);

      Settings::StageSetting stage;
      stage.mScaling      = this->mSolarSystem->getObserver().getAnchorScale();
      stage.mBookmarkName = bookmark.mName;
      mPluginSettings->mStageSettings.push_back(stage);

      logger().info("Recorded Checkpoint {}.", bookmark.mName);
    }

  } else {

    if (mEnableCOGMeasurement) {
      auto translation =
          GetVistaSystem()
              ->GetPlatformFor(GetVistaSystem()->GetDisplayManager()->GetDisplaySystem())
              ->GetPlatformNode()
              ->GetTranslation();
      resultsLogger().info("{}: COG {} {} {}",
          mPluginSettings->mStageSettings[mStageIdx].mBookmarkName.get(), -translation[0],
          -translation[1], -translation[2]);
    }

    // check if current stage is normal checkpoint
    if (mPluginSettings->mStageSettings[mStageIdx].mType.get() == Plugin::StageType::eCheckpoint) {
      // check distance to CP
      glm::dvec3 vecToObserver = mStages[mStageIdx % mStages.size()].mAnchor->getRelativePosition(
          mTimeControl->pSimulationTime.get(), mSolarSystem->getObserver());
      if (glm::length(vecToObserver) < mPluginSettings->mStageSettings[mStageIdx].mScaling.get()) {
        // go to next stage
        resultsLogger().info("{}: Passed Checkpoint",
            mPluginSettings->mStageSettings[mStageIdx].mBookmarkName.get());
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

  mGuiManager->removeSettingsSection("User Study");
  mGuiManager->getGui()->unregisterCallback("userStudy.setRecordingInterval");
  mGuiManager->getGui()->unregisterCallback("userStudy.deleteAllCheckpoints");
  mGuiManager->getGui()->unregisterCallback("userStudy.setEnableRecording");
  mGuiManager->getGui()->unregisterCallback("userStudy.setEnableCOGMeasurement");

  logger().info("Unloading done.");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::onLoad() {

  unload();
  mStageIdx = 0;

  // Read settings from JSON
  from_json(mAllSettings->mPlugins.at("csp-user-study"), *mPluginSettings);

  // Get scenegraph to init stages
  VistaSceneGraph* pSG = GetVistaSystem()->GetGraphicsManager()->GetSceneGraph();
  // Init stages
  for (std::size_t i = 0; i < mStages.size(); i++) {
    Stage& stage = mStages[i];
    // Create and register anchor
    stage.mAnchor =
        std::make_shared<cs::scene::CelestialAnchorNode>(pSG->GetRoot(), pSG->GetNodeBridge());
    mSolarSystem->registerAnchor(stage.mAnchor);

    // Create and setup gui area
    stage.mGuiArea = std::make_unique<cs::gui::WorldSpaceGuiArea>(720, 720);
    stage.mGuiArea->setUseLinearDepthBuffer(true);

    // Create transform node
    stage.mTransform =
        std::unique_ptr<VistaTransformNode>(pSG->NewTransformNode(stage.mAnchor.get()));

    // Create gui node
    stage.mGuiNode = std::unique_ptr<VistaOpenGLNode>(
        pSG->NewOpenGLNode(stage.mTransform.get(), stage.mGuiArea.get()));

    // Register selectable
    mInputManager->registerSelectable(stage.mGuiNode.get());

    // Create gui item & attach it to gui area
    stage.mGuiItem = std::make_unique<cs::gui::GuiItem>(
        "file://{csp-user-study-cp}../share/resources/gui/user-study-stage.html");
    stage.mGuiArea->addItem(stage.mGuiItem.get());
    stage.mGuiItem->waitForFinishedLoading();
    stage.mGuiItem->setZoomFactor(2);

    // register callbacks
    stage.mGuiItem->registerCallback("setFMS", "Callback to get slider value",
        std::function([this](double value) { mCurrentFMS = static_cast<uint32_t>(value); }));
    stage.mGuiItem->registerCallback(
        "confirmFMS", "Call this to submit the FMS rating", std::function([this]() {
          resultsLogger().info("{}: FMS: {}",
              mPluginSettings->mStageSettings[mStageIdx].mBookmarkName.get(), mCurrentFMS.get());
          advanceStage();
        }));
    stage.mGuiItem->registerCallback(
        "loadScenario", "Call this to load a new scenario", std::function([this](std::string path) {
          resultsLogger().info("Loading Scenario at " + path);
          mGuiManager->getGui()->callJavascript("CosmoScout.callbacks.core.load", path);
        }));
    stage.mGuiItem->registerCallback("setEnableCOGMeasurement",
        "Enables or disables center of gravity recording.", std::function([this](bool enable) {
          mEnableCOGMeasurement = enable;

          if (!enable) {
            advanceStage();
          }
        }));

    setupStage(i);
  }

  // register FMS callback part afterwards
  mCurrentFMS.connectAndTouch([this](uint32_t value) {
    for (std::size_t i = 0; i < mStages.size(); i++) {
      mStages[i].mGuiItem->callJavascript("CosmoScout.gui.setSliderValue", "setFMS", false, value);
    }
  });

  updateStages();

  // Mark start of scenario in results log
  resultsLogger().info("Scenario started");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::setupStage(std::size_t stageIdx) {
  auto const& settings = mPluginSettings->mStageSettings[stageIdx];

  // Fetch stage at Index
  Stage& stage = mStages[(stageIdx) % mStages.size()];
  // Fetch bookmark for position
  std::optional<cs::core::Settings::Bookmark> bookmark =
      getBookmarkByName(settings.mBookmarkName.get());
  if (bookmark.has_value()) {
    cs::core::Settings::Bookmark b = bookmark.value();
    // Move anchor to bookmark position
    stage.mAnchor->setCenterName(b.mLocation->mCenter);
    stage.mAnchor->setFrameName(b.mLocation->mFrame);
    if (b.mLocation->mPosition.has_value()) {
      stage.mAnchor->setAnchorPosition(b.mLocation->mPosition.value());
    }
    if (b.mLocation->mPosition.has_value()) {
      stage.mAnchor->setAnchorRotation(b.mLocation->mRotation.value());
    }

    // Add Scaling factor
    const float checkPointScale = 2.f;
    stage.mTransform->SetScale(
        settings.mScaling.get() * checkPointScale, settings.mScaling.get() * checkPointScale, 1.0F);

    // Set webview according to type
    switch (settings.mType.get()) {
    case StageType::eCheckpoint: {
      stage.mGuiItem->callJavascript("setCP");
      break;
    }
    case StageType::eRequestFMS: {
      stage.mGuiItem->callJavascript("setFMS");
      break;
    }
    case StageType::eRequestCOG: {
      stage.mGuiItem->callJavascript("setCOG");
      break;
    }
    case StageType::eSwitchScenario: {
      std::string html = "";
      for (Plugin::Settings::Scenario& scenario : mPluginSettings->mOtherScenarios) {
        html += "<input class=\"btn\" type=\"button\" value=\"" + scenario.mName.get() +
                "\" onclick=\"window.callNative('loadScenario', '" + scenario.mPath.get() +
                "')\">\n";
      }
      stage.mGuiItem->callJavascript("setCHS", html);
      break;
    }
    default: {
      logger().error(
          "Unrecognized stage type when setting up Stage! stage type: {}", settings.mType.get());
      break;
    }
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::unload() {
  VistaSceneGraph* pSG = GetVistaSystem()->GetGraphicsManager()->GetSceneGraph();
  for (Stage& stage : mStages) {
    // skip unload if Stage is empty
    if (stage.mAnchor == nullptr) {
      break;
    }
    // unregister callbacks
    stage.mGuiItem->unregisterCallback("setFMS");
    stage.mGuiItem->unregisterCallback("confirmFMS");
    stage.mGuiItem->unregisterCallback("loadScenario");
    // disconnect from scene graph
    pSG->GetRoot()->DisconnectChild(stage.mAnchor.get());
    stage.mAnchor->DisconnectChild(stage.mTransform.get());
    stage.mTransform->DisconnectChild(stage.mGuiNode.get());
    // unregister anchor
    mSolarSystem->unregisterAnchor(stage.mAnchor);
    // unregister selectable
    mInputManager->unregisterSelectable(stage.mGuiNode.get());

    stage = {};
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

void Plugin::updateStages() {

  // Set css classes of all visible stages. The item at i==0 is the current stage, and
  // i==mStages.size()-1 is the most distant stage.
  for (std::size_t i = 0; i < mStages.size(); i++) {
    auto stageIdx = (mStageIdx + i) % mStages.size();
    if (mStageIdx + i < mPluginSettings->mStageSettings.size()) {
      mStages[stageIdx].mGuiItem->callJavascript("setBodyClass", "stage" + std::to_string(i));
    } else {
      mStages[stageIdx].mGuiItem->callJavascript("setBodyClass", "hidden");
    }

    // Make only current stage interactive.
    mStages[stageIdx].mGuiItem->setIsInteractive(i == 0);

    // Ensure that the chaecpoints are drawn back-to-front.
    std::size_t sortKey =
        static_cast<std::size_t>(cs::utils::DrawOrder::eTransparentItems) + mStages.size() - i;
    VistaOpenSGMaterialTools::SetSortKeyOnSubtree(
        mStages[stageIdx].mGuiNode.get(), static_cast<int>(sortKey));
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::advanceStage() {

  // Advance the current stage index.
  mStageIdx = std::min(mStageIdx + 1, mPluginSettings->mStageSettings.size() - 1);

  // Setup the stage which becomes visible next.
  std::size_t newlyVisibleIdx = mStageIdx + mStages.size() - 1;
  if (newlyVisibleIdx < mPluginSettings->mStageSettings.size()) {
    setupStage(newlyVisibleIdx);
  }

  updateStages();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::userstudy
