////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "GuiManager.hpp"

#include <GL/freeglut.h>

#include "../cs-utils/filesystem.hpp"
#include "../cs-utils/utils.hpp"
#include "InputManager.hpp"
#include "cs-version.hpp"
#include "tools/Tool.hpp"

#include <VistaKernel/DisplayManager/GlutWindowImp/VistaGlutWindowingToolkit.h>
#include <VistaKernel/DisplayManager/VistaDisplayManager.h>
#include <VistaKernel/DisplayManager/VistaDisplaySystem.h>
#include <VistaKernel/DisplayManager/VistaViewportResizeToProjectionAdapter.h>
#include <VistaKernel/GraphicsManager/VistaOpenGLNode.h>
#include <VistaKernel/GraphicsManager/VistaSceneGraph.h>
#include <VistaKernel/GraphicsManager/VistaTransformNode.h>
#include <VistaKernel/InteractionManager/VistaUserPlatform.h>
#include <VistaKernel/Stuff/VistaFramerateDisplay.h>
#include <VistaKernel/VistaFrameLoop.h>
#include <VistaKernel/VistaSystem.h>
#include <VistaKernelOpenSGExt/VistaOpenSGMaterialTools.h>
#include <fstream>
#include <spdlog/spdlog.h>
#include <utility>

namespace cs::core {

////////////////////////////////////////////////////////////////////////////////////////////////////

GuiManager::GuiManager(std::shared_ptr<const Settings> const& settings,
    std::shared_ptr<InputManager> pInputManager, std::shared_ptr<utils::FrameTimings> pFrameTimings)
    : mInputManager(std::move(pInputManager))
    , mFrameTimings(std::move(pFrameTimings)) {

  // Tell the user what's going on.
  spdlog::debug("Creating GuiManager.");

  // Initialize the Chromium Embedded Framework.
  gui::init();

  // Update the main viewport when the window is resized.
  VistaViewport* pViewport(GetVistaSystem()->GetDisplayManager()->GetViewports().begin()->second);
  mViewportUpdater = std::make_unique<VistaViewportResizeToProjectionAdapter>(pViewport);
  mViewportUpdater->SetUpdateMode(VistaViewportResizeToProjectionAdapter::MAINTAIN_HORIZONTAL_FOV);

  // Hide the user interface when ESC is pressed.
  mInputManager->sOnEscapePressed.connect([this]() { toggleGui(); });

  // Create GuiAreas and attach them to the SceneGraph ---------------------------------------------

  // The global GUI is drawn in world-space, the local GUI is drawn in screen-space.
  auto* pSG          = GetVistaSystem()->GetGraphicsManager()->GetSceneGraph();
  mLocalGuiTransform = pSG->NewTransformNode(pSG->GetRoot());

  // The global GUI area is only created when the according settings key was specified.
  if (settings->mGui) {
    auto* platform = GetVistaSystem()
                         ->GetPlatformFor(GetVistaSystem()->GetDisplayManager()->GetDisplaySystem())
                         ->GetPlatformNode();
    mGlobalGuiTransform = pSG->NewTransformNode(platform);

    mGlobalGuiTransform->Scale(static_cast<float>(settings->mGui->mWidthMeter),
        static_cast<float>(settings->mGui->mHeightMeter), 1.0);
    mGlobalGuiTransform->Rotate(
        VistaAxisAndAngle(VistaVector3D(1, 0, 0), static_cast<float>(settings->mGui->mRotX)));
    mGlobalGuiTransform->Rotate(
        VistaAxisAndAngle(VistaVector3D(0, 1, 0), static_cast<float>(settings->mGui->mRotY)));
    mGlobalGuiTransform->Rotate(
        VistaAxisAndAngle(VistaVector3D(0, 0, 1), static_cast<float>(settings->mGui->mRotZ)));
    mGlobalGuiTransform->Translate(static_cast<float>(settings->mGui->mPosXMeter),
        static_cast<float>(settings->mGui->mPosYMeter),
        static_cast<float>(settings->mGui->mPosZMeter));

    // Create the global GUI area.
    mGlobalGuiArea = std::make_unique<gui::WorldSpaceGuiArea>(
        settings->mGui->mWidthPixel, settings->mGui->mHeightPixel);
    mGlobalGuiArea->setUseLinearDepthBuffer(true);
  }

  // Create the local GUI area.
  mLocalGuiArea = std::make_unique<gui::ScreenSpaceGuiArea>(pViewport);

  // Make sure that the GUI is drawn at the correct position in the draw order.
  mLocalGuiOpenGLnode = pSG->NewOpenGLNode(mLocalGuiTransform, mLocalGuiArea.get());
  VistaOpenSGMaterialTools::SetSortKeyOnSubtree(
      mLocalGuiOpenGLnode, static_cast<int>(utils::DrawOrder::eGui));

  // Make the local GuiArea receive input events.
  mInputManager->registerSelectable(mLocalGuiArea.get());

  if (mGlobalGuiArea) {
    // Make sure that the GUI is drawn at the correct position in the draw order.
    mGlobalGuiOpenGLnode = pSG->NewOpenGLNode(mGlobalGuiTransform, mGlobalGuiArea.get());
    VistaOpenSGMaterialTools::SetSortKeyOnSubtree(
        mGlobalGuiTransform, static_cast<int>(utils::DrawOrder::eGui));

    // Make the global GuiArea receive input events.
    mInputManager->registerSelectable(mGlobalGuiOpenGLnode);
  }

  // Now create the actual Gui and add it to the previously created GuiAreas ----------------
  mCosmoScoutGui = std::make_unique<gui::GuiItem>("file://../share/resources/gui/cosmoscout.html");
  mStatistics    = std::make_unique<gui::GuiItem>("file://../share/resources/gui/statistics.html");

  // Except for mStatistics, all GuiItems are attached to the global world-space GuiArea if it is
  // available. If not, they are added to the local screen-space GuiArea.
  if (mGlobalGuiArea) {
    mGlobalGuiArea->addItem(mCosmoScoutGui.get());
  } else {
    mLocalGuiArea->addItem(mCosmoScoutGui.get());
  }

  mLocalGuiArea->addItem(mStatistics.get());

  // Configure attributes of the loading screen. Per default, GuiItems are drawn full-screen in
  // their GuiAreas.
  // mLoadingScreen->setIsInteractive(false);

  mCosmoScoutGui->setRelSizeX(1.F);
  mCosmoScoutGui->setRelSizeY(1.F);
  mCosmoScoutGui->setRelPositionX(0.5F);
  mCosmoScoutGui->setRelPositionY(0.5F);
  mCosmoScoutGui->setCursorChangeCallback([](gui::Cursor c) { setCursor(c); });

  // Configure the positioning and attributes of the statistics.
  mStatistics->setSizeX(600);    // NOLINT
  mStatistics->setSizeY(320);    // NOLINT
  mStatistics->setOffsetX(-300); // NOLINT
  mStatistics->setOffsetY(500);  // NOLINT
  mStatistics->setRelPositionY(0.F);
  mStatistics->setRelPositionX(1.F);
  mStatistics->setIsInteractive(false);
  mStatistics->setCanScroll(false);

  // Now we will call some JavaScript methods - so we have to wait until the GuiItems have been
  // fully loaded.
  mCosmoScoutGui->waitForFinishedLoading();

  // Create a string which contains the current version number of CosmoScout VR. This string is then
  // shown on the loading screen.
  std::string version("v" + CS_PROJECT_VERSION);

  if (CS_GIT_BRANCH == "HEAD") {
    version += " (@" + CS_GIT_COMMIT_HASH + ")";
  } else if (!CS_GIT_BRANCH.empty()) {
    version += " (" + CS_GIT_BRANCH;
    if (!CS_GIT_COMMIT_HASH.empty()) {
      version += " @" + CS_GIT_COMMIT_HASH;
    }
    version += ")";
  }

  mCosmoScoutGui->callJavascript("CosmoScout.loadingScreen.setVersion", version);

  // Set settings for the time Navigation
  mCosmoScoutGui->callJavascript(
      "CosmoScout.timeline.setTimelineRange", settings->mMinDate, settings->mMaxDate);

  for (const auto& mEvent : settings->mEvents) {
    std::string planet;
    std::string place;
    if (mEvent.mLocation.has_value()) {
      planet = mEvent.mLocation.value().mPlanet;
      place  = mEvent.mLocation.value().mPlace;
    }
    addEventToTimenavigationBar(mEvent.mStart, mEvent.mEnd, mEvent.mId, mEvent.mContent,
        mEvent.mStyle, mEvent.mDescription, planet, place);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

GuiManager::~GuiManager() {
  try {
    // Tell the user what's going on.
    spdlog::debug("Deleting GuiManager.");
  } catch (...) {}

  mInputManager->unregisterSelectable(mLocalGuiOpenGLnode);

  if (mGlobalGuiOpenGLnode) {
    mInputManager->unregisterSelectable(mGlobalGuiOpenGLnode);
  }

  // Free resources acquired by the Chromium Embedded Framework.
  gui::cleanUp();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void GuiManager::setCursor(gui::Cursor cursor) {
  auto* windowingToolkit = dynamic_cast<VistaGlutWindowingToolkit*>(
      GetVistaSystem()->GetDisplayManager()->GetWindowingToolkit());

  int glutCursor = GLUT_CURSOR_LEFT_ARROW;

  switch (cursor) {
  case gui::Cursor::ePointer:
    glutCursor = GLUT_CURSOR_LEFT_ARROW;
    break;
  case gui::Cursor::eCross:
    glutCursor = GLUT_CURSOR_CROSSHAIR;
    break;
  case gui::Cursor::eHelp:
    glutCursor = GLUT_CURSOR_HELP;
    break;
  case gui::Cursor::eWait:
    glutCursor = GLUT_CURSOR_WAIT;
    break;
  case gui::Cursor::eIbeam:
    glutCursor = GLUT_CURSOR_TEXT;
    break;
  case gui::Cursor::eHand:
    glutCursor = GLUT_CURSOR_INFO;
    break;
  default:
    break;
  }

  if (windowingToolkit) {
    for (auto const& window : GetVistaSystem()->GetDisplayManager()->GetWindows()) {
      windowingToolkit->SetCursor(window.second, glutCursor);
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void GuiManager::showNotification(std::string const& sTitle, std::string const& sText,
    std::string const& sIcon, std::string const& sFlyToOnClick) const {
  mCosmoScoutGui->callJavascript(
      "CosmoScout.notifications.print", sTitle, sText, sIcon, sFlyToOnClick);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

gui::GuiItem* GuiManager::getGui() const {
  return mCosmoScoutGui.get();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

gui::GuiItem* GuiManager::getStatistics() const {
  return mStatistics.get();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void GuiManager::enableLoadingScreen(bool enable) {
  mCosmoScoutGui->callJavascript("CosmoScout.loadingScreen.setLoading", enable);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void GuiManager::setLoadingScreenStatus(std::string const& sStatus) const {
  mCosmoScoutGui->callJavascript("CosmoScout.loadingScreen.setStatus", sStatus);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void GuiManager::setLoadingScreenProgress(float percent, bool animate) const {
  mCosmoScoutGui->callJavascript("CosmoScout.loadingScreen.setProgress", percent, animate);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void GuiManager::showGui() {
  if (mGlobalGuiTransform) {
    mGlobalGuiTransform->SetIsEnabled(true);
  }
  mLocalGuiTransform->SetIsEnabled(true);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void GuiManager::hideGui() {
  if (mGlobalGuiTransform) {
    mGlobalGuiTransform->SetIsEnabled(false);
  }
  mLocalGuiTransform->SetIsEnabled(false);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void GuiManager::toggleGui() {
  if (mGlobalGuiTransform) {
    mGlobalGuiTransform->SetIsEnabled(!mGlobalGuiTransform->GetIsEnabled());
  }
  mLocalGuiTransform->SetIsEnabled(!mLocalGuiTransform->GetIsEnabled());
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void GuiManager::update() {

  // If frame timings are enabled, collect the data and send it to the statistics GuiItem.
  mStatistics->setIsEnabled(mFrameTimings->pEnableMeasurements.get());
  if (mFrameTimings->pEnableMeasurements.get()) {
    std::string json("{");
    for (auto const& timings : mFrameTimings->getCalculatedQueryResults()) {
      uint64_t timeGPU(timings.second.mGPUTime);
      uint64_t timeCPU(timings.second.mCPUTime);

      uint64_t const waitNanos = 100000;
      if (timeGPU > waitNanos || timeCPU > waitNanos) {
        json += "\"" + timings.first + "\":[" + std::to_string(timeGPU) + "," +
                std::to_string(timeCPU) + "],";
      }
    }
    json.back() = '}';

    if (json.length() <= 1) {
      json = "{}";
    }

    mStatistics->callJavascript(
        "CosmoScout.statistics.setData", json, GetVistaSystem()->GetFrameLoop()->GetFrameRate());
  }

  // Update all entities of the Chromium Embedded Framework.
  gui::update();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void GuiManager::addPluginTabToSideBar(
    std::string const& name, std::string const& icon, std::string const& content) {
  mCosmoScoutGui->callJavascript("CosmoScout.sidebar.addPluginTab", name, icon, content);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void GuiManager::addPluginTabToSideBarFromHTML(
    std::string const& name, std::string const& icon, std::string const& htmlFile) {
  std::string content = utils::filesystem::loadToString(htmlFile);
  addPluginTabToSideBar(name, icon, content);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void GuiManager::addSettingsSectionToSideBar(
    std::string const& name, std::string const& icon, std::string const& content) {
  mCosmoScoutGui->callJavascript("CosmoScout.sidebar.addSettingsSection", name, icon, content);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void GuiManager::addSettingsSectionToSideBarFromHTML(
    std::string const& name, std::string const& icon, std::string const& htmlFile) {
  std::string content = utils::filesystem::loadToString(htmlFile);
  addSettingsSectionToSideBar(name, icon, content);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void GuiManager::removePluginTab(std::string const& name) {
  mCosmoScoutGui->callJavascript("CosmoScout.sidebar.removePluginTab", name);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void GuiManager::removeSettingsSection(std::string const& name) {
  mCosmoScoutGui->callJavascript("CosmoScout.sidebar.removeSettingsSection", name);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void GuiManager::addScriptToGui(std::string const& src) {
  mCosmoScoutGui->executeJavascript(src);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void GuiManager::addScriptToGuiFromJS(std::string const& jsFile) {
  std::string content = utils::filesystem::loadToString(jsFile);
  addScriptToGui(content);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void GuiManager::addHtmlToGui(std::string const& id, std::string const& src) {
  std::string content = utils::filesystem::loadToString(src);
  mCosmoScoutGui->callJavascript("CosmoScout.gui.registerHtml", id, content);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void GuiManager::addCssToGui(const std::string& fileName) {
  mCosmoScoutGui->callJavascript("CosmoScout.gui.registerCss", fileName);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void GuiManager::addEventToTimenavigationBar(std::string const& start,
    std::optional<std::string> const& end, std::string const& id, std::string const& content,
    std::optional<std::string> const& style, std::string const& description,
    std::string const& planet, std::string const& place) {
  mCosmoScoutGui->callJavascript("CosmoScout.timeline.addItem", start, end.value_or(""), id,
      content, style.value_or(""), description, planet, place);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cs::core
