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

namespace cs::core {

////////////////////////////////////////////////////////////////////////////////////////////////////

GuiManager::GuiManager(std::shared_ptr<const Settings> const& settings,
    std::shared_ptr<InputManager> const&                      pInputManager,
    std::shared_ptr<utils::FrameTimings> const&               pFrameTimings)
    : mInputManager(pInputManager)
    , mFrameTimings(pFrameTimings) {

  std::cout << "Loading: GuiManager" << std::endl;

  // Initialize the Chromium Embedded Framework.
  gui::init();

  // Update the main viewport when the window is resized.
  VistaViewport* pViewport(GetVistaSystem()->GetDisplayManager()->GetViewports().begin()->second);
  mViewportUpdater = new VistaViewportResizeToProjectionAdapter(pViewport);
  mViewportUpdater->SetUpdateMode(VistaViewportResizeToProjectionAdapter::MAINTAIN_HORIZONTAL_FOV);

  // Hide the user interface when ESC is pressed.
  mInputManager->sOnEscapePressed.connect([this]() { toggleGui(); });

  // Create GuiAreas and attach them to the SceneGraph ---------------------------------------------

  // The global GUI is drawn in world-space, the local GUI is drawn in screen-space.
  auto pSG           = GetVistaSystem()->GetGraphicsManager()->GetSceneGraph();
  mLocalGuiTransform = pSG->NewTransformNode(pSG->GetRoot());

  // The global GUI area is only created when the according settings key was specified.
  if (settings->mGui) {
    auto platform = GetVistaSystem()
                        ->GetPlatformFor(GetVistaSystem()->GetDisplayManager()->GetDisplaySystem())
                        ->GetPlatformNode();
    mGlobalGuiTransform = pSG->NewTransformNode(platform);

    mGlobalGuiTransform->Scale(
        (float)settings->mGui->mWidthMeter, (float)settings->mGui->mHeightMeter, 1.0);
    mGlobalGuiTransform->Rotate(
        VistaAxisAndAngle(VistaVector3D(1, 0, 0), (float)settings->mGui->mRotX));
    mGlobalGuiTransform->Rotate(
        VistaAxisAndAngle(VistaVector3D(0, 1, 0), (float)settings->mGui->mRotY));
    mGlobalGuiTransform->Rotate(
        VistaAxisAndAngle(VistaVector3D(0, 0, 1), (float)settings->mGui->mRotZ));
    mGlobalGuiTransform->Translate((float)settings->mGui->mPosXMeter,
        (float)settings->mGui->mPosYMeter, (float)settings->mGui->mPosZMeter);

    // Create the global GUI area.
    mGlobalGuiArea =
        new gui::WorldSpaceGuiArea(settings->mGui->mWidthPixel, settings->mGui->mHeightPixel);
    mGlobalGuiArea->setUseLinearDepthBuffer(true);
  }

  // Create the local GUI area.
  mLocalGuiArea = new gui::ScreenSpaceGuiArea(pViewport);

  // Make sure that the GUI is drawn at the correct position in the draw order.
  mLocalGuiOpenGLnode = pSG->NewOpenGLNode(mLocalGuiTransform, mLocalGuiArea);
  VistaOpenSGMaterialTools::SetSortKeyOnSubtree(
      mLocalGuiOpenGLnode, static_cast<int>(utils::DrawOrder::eGui));

  // Make the local GuiArea receive input events.
  mInputManager->registerSelectable(mLocalGuiArea);

  if (mGlobalGuiArea) {
    // Make sure that the GUI is drawn at the correct position in the draw order.
    mGlobalGuiOpenGLnode = pSG->NewOpenGLNode(mGlobalGuiTransform, mGlobalGuiArea);
    VistaOpenSGMaterialTools::SetSortKeyOnSubtree(
        mGlobalGuiTransform, static_cast<int>(utils::DrawOrder::eGui));

    // Make the global GuiArea receive input events.
    mInputManager->registerSelectable(mGlobalGuiOpenGLnode);
  }

  // Now create the actual GuiItems and add them to the previously created GuiAreas ----------------

  mLoadingScreen = new gui::GuiItem("file://../share/resources/gui/loading_screen.html");
  mSideBar       = new gui::GuiItem("file://../share/resources/gui/sidebar.html");
  mStatusBar     = new gui::GuiItem("file://../share/resources/gui/statusbar.html");
  mNotifications = new gui::GuiItem("file://../share/resources/gui/notifications.html");
  mLogo          = new gui::GuiItem("file://../share/resources/gui/logo.html");
  mStatistics    = new gui::GuiItem("file://../share/resources/gui/statistics.html");
  mTimeline      = new gui::GuiItem("file://../share/resources/gui/timeline.html");

  // Except for mStatistics, all GuiItems are attached to the global world-space GuiArea if it is
  // available. If not, they are added to the local screen-space GuiArea.
  if (mGlobalGuiArea) {
    mGlobalGuiArea->addItem(mLogo);
    mGlobalGuiArea->addItem(mNotifications);
    mGlobalGuiArea->addItem(mStatusBar);
    mGlobalGuiArea->addItem(mSideBar);
    mGlobalGuiArea->addItem(mTimeline);
    mGlobalGuiArea->addItem(mLoadingScreen);
  } else {
    mLocalGuiArea->addItem(mLogo);
    mLocalGuiArea->addItem(mNotifications);
    mLocalGuiArea->addItem(mStatusBar);
    mLocalGuiArea->addItem(mSideBar);
    mLocalGuiArea->addItem(mTimeline);
    mLocalGuiArea->addItem(mLoadingScreen);
  }

  mLocalGuiArea->addItem(mStatistics);

  // Configure attributes of the loading screen. Per default, GuiItems are drawn full-screen in
  // their GuiAreas.
  mLoadingScreen->setIsInteractive(false);

  // Configure the positioning and attributes of the side-bar.
  mSideBar->setSizeX(500);
  mSideBar->setRelSizeY(1.f);
  mSideBar->setRelPositionY(1.f);
  mSideBar->setRelPositionX(0.f);
  mSideBar->setOffsetX(250);
  mSideBar->setRelOffsetY(-0.5f);
  mSideBar->setCursorChangeCallback([this](gui::Cursor c) { setCursor(c); });

  mStatusBar->setRelSizeX(1.f);
  mStatusBar->setSizeY(80);
  mStatusBar->setRelPositionX(0.5);
  mStatusBar->setRelPositionY(1.f);
  mStatusBar->setCursorChangeCallback([this](gui::Cursor c) { setCursor(c); });

  mTimeline->setRelSizeX(1.f);
  mTimeline->setSizeY(644);
  mTimeline->setRelPositionX(0.5);
  mTimeline->setRelPositionY(0);
  mTimeline->setOffsetY(322);
  mTimeline->setCursorChangeCallback([this](gui::Cursor c) { setCursor(c); });

  // Configure the positioning and attributes of the notifications.
  mNotifications->setSizeX(420);
  mNotifications->setSizeY(320);
  mNotifications->setRelPositionY(0.f);
  mNotifications->setRelPositionX(1.f);
  mNotifications->setOffsetX(-210);
  mNotifications->setOffsetY(250);
  mNotifications->setIsInteractive(false);

  // Configure the positioning and attributes of the logo.
  mLogo->setSizeX(120);
  mLogo->setSizeY(100);
  mLogo->setRelPositionY(1.f);
  mLogo->setRelPositionX(1.f);
  mLogo->setOffsetX(-60);
  mLogo->setOffsetY(-50);
  mLogo->setIsInteractive(false);

  // Configure the positioning and attributes of the statistics.
  mStatistics->setSizeX(1200);
  mStatistics->setSizeY(300);
  mStatistics->setOffsetX(-600);
  mStatistics->setOffsetY(300);
  mStatistics->setRelPositionY(0.f);
  mStatistics->setRelPositionX(1.f);
  mStatistics->setIsInteractive(false);

  // Now we will call some JavaScript methods - so we have to wait until the GuiItems have been
  // fully loaded.
  mSideBar->waitForFinishedLoading();
  mStatusBar->waitForFinishedLoading();
  mTimeline->waitForFinishedLoading();
  mNotifications->waitForFinishedLoading();
  mLoadingScreen->waitForFinishedLoading();

  // Create a string which contains the current version number of CosmoScout VR. This string is then
  // shown on the loading screen.
  std::string version("v" + CS_PROJECT_VERSION);

  if (CS_GIT_BRANCH == "HEAD") {
    version += " (@" + CS_GIT_COMMIT_HASH + ")";
  } else if (CS_GIT_BRANCH != "") {
    version += " (" + CS_GIT_BRANCH;
    if (CS_GIT_COMMIT_HASH != "") {
      version += " @" + CS_GIT_COMMIT_HASH;
    }
    version += ")";
  }

  mLoadingScreen->callJavascript("set_version", version);

  mLoadingScreen->registerCallback("finished_fadeout", [this]() {
    if (mGlobalGuiArea) {
      mGlobalGuiArea->removeItem(mLoadingScreen);
    } else {
      mLocalGuiArea->removeItem(mLoadingScreen);
    }
  });

  // Set settings for the time Navigation
  mTimeline->callJavascript("set_timeline_range", settings->mMinDate, settings->mMaxDate);

  for (int i = 0; i < settings->mEvents.size(); i++) {
    std::string planet = "";
    std::string place  = "";
    if (settings->mEvents.at(i).mLocation.has_value()) {
      planet = settings->mEvents.at(i).mLocation.value().mPlanet;
      place  = settings->mEvents.at(i).mLocation.value().mPlace;
    }
    addEventToTimenavigationBar(settings->mEvents.at(i).mStart, settings->mEvents.at(i).mEnd,
        settings->mEvents.at(i).mId, settings->mEvents.at(i).mContent,
        settings->mEvents.at(i).mStyle, settings->mEvents.at(i).mDescription, planet, place);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

GuiManager::~GuiManager() {
  delete mSideBar;
  delete mStatusBar;
  delete mNotifications;
  delete mLogo;
  delete mGlobalGuiArea;
  delete mViewportUpdater;
  delete mTimeline;

  mInputManager->unregisterSelectable(mLocalGuiOpenGLnode);

  if (mGlobalGuiOpenGLnode) {
    mInputManager->unregisterSelectable(mGlobalGuiOpenGLnode);
  }

  // Free resources acquired by the Chromium Embedded Framework.
  gui::cleanUp();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void GuiManager::setCursor(gui::Cursor cursor) {
  auto windowingToolkit = dynamic_cast<VistaGlutWindowingToolkit*>(
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
  mNotifications->callJavascript("print_notification", sTitle, sText, sIcon, sFlyToOnClick);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

gui::GuiItem* GuiManager::getSideBar() const {
  return mSideBar;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

gui::GuiItem* GuiManager::getStatusBar() const {
  return mStatusBar;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

gui::GuiItem* GuiManager::getTimeline() const {
  return mTimeline;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

gui::GuiItem* GuiManager::getStatistics() const {
  return mStatistics;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

gui::GuiItem* GuiManager::getLogo() const {
  return mLogo;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void GuiManager::enableLoadingScreen(bool enable) {
  mLoadingScreen->callJavascript("set_loading", enable);

  if (enable) {
    if (mGlobalGuiArea) {
      mGlobalGuiArea->addItem(mLoadingScreen);
    } else {
      mLocalGuiArea->addItem(mLoadingScreen);
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void GuiManager::setLoadingScreenStatus(std::string const& sStatus) const {
  mLoadingScreen->callJavascript("set_status", sStatus);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void GuiManager::setLoadingScreenProgress(float percent, bool animate) const {
  mLoadingScreen->callJavascript("set_progress", percent, animate);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void GuiManager::registerTool(std::shared_ptr<tools::Tool> const& tool) {
  mTools.push_back(tool);
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

  // Update all registered tools. If the pShouldDelete property is set, the Tool is removed from the
  // list.
  for (auto it = mTools.begin(); it != mTools.end();) {
    if ((*it)->pShouldDelete.get()) {
      it = mTools.erase(it);
    } else {
      (*it)->update();
      ++it;
    }
  }

  // If frame timings are enabled, collect the data and send it to the statistics GuiItem.
  if (mFrameTimings->pEnableMeasurements.get()) {
    std::string json("{");
    for (auto const& timings : mFrameTimings->getCalculatedQueryResults()) {
      uint64_t timeGPU(timings.second.mGPUTime), timeCPU(timings.second.mCPUTime);

      if (timeGPU > 100000 || timeCPU > 100000) {
        json += "\"" + timings.first + "\":[" + std::to_string(timeGPU) + "," +
                std::to_string(timeCPU) + "],";
      }
    }
    json.back() = '}';

    if (json.length() <= 1) {
      json = "{}";
    }

    mStatistics->callJavascript("set_data", json, GetVistaSystem()->GetFrameLoop()->GetFrameRate());
  }

  // Update all entities of the Chromium Embedded Framework.
  gui::update();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void GuiManager::addPluginTabToSideBar(
    std::string const& name, std::string const& icon, std::string const& content) {
  mSideBar->callJavascript("addPluginTab", name, icon, content);
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
  mSideBar->callJavascript("addSettingsSection", name, icon, content);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void GuiManager::addSettingsSectionToSideBarFromHTML(
    std::string const& name, std::string const& icon, std::string const& htmlFile) {
  std::string content = utils::filesystem::loadToString(htmlFile);
  addSettingsSectionToSideBar(name, icon, content);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void GuiManager::addScriptToSideBar(std::string const& src) {
  mSideBar->executeJavascript(src);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void GuiManager::addScriptToSideBarFromJS(std::string const& jsFile) {
  std::string content = utils::filesystem::loadToString(jsFile);
  addScriptToSideBar(content);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void GuiManager::addEventToTimenavigationBar(std::string start, std::optional<std::string> end,
    std::string id, std::string content, std::optional<std::string> style, std::string description,
    std::string planet, std::string place) {
  mTimeline->callJavascript("add_item", start, end.value_or(""), id, content, style.value_or(""),
      description, planet, place);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cs::core
