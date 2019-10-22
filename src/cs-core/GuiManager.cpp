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

  gui::init();

  mInputManager->sOnEscapePressed.connect([this]() { toggleGui(); });

  auto pSG = GetVistaSystem()->GetGraphicsManager()->GetSceneGraph();

  // Attach gui to a common group node.
  auto platform = GetVistaSystem()
                      ->GetPlatformFor(GetVistaSystem()->GetDisplayManager()->GetDisplaySystem())
                      ->GetPlatformNode();
  mGlobalGuiTransform = pSG->NewTransformNode(platform);
  mLocalGuiTransform  = pSG->NewTransformNode(pSG->GetRoot());

  // Create gui areas.

  if (settings->mGui) {
    mGlobalGuiArea =
        new gui::WorldSpaceGuiArea(settings->mGui->mWidthPixel, settings->mGui->mHeightPixel);
    mGlobalGuiArea->setUseLinearDepthBuffer(true);
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
  }

  VistaViewport* pViewport(GetVistaSystem()->GetDisplayManager()->GetViewports().begin()->second);

  mLocalGuiArea = new gui::ScreenSpaceGuiArea(pViewport);

  mViewportUpdater = new VistaViewportResizeToProjectionAdapter(pViewport);
  mViewportUpdater->SetUpdateMode(VistaViewportResizeToProjectionAdapter::MAINTAIN_HORIZONTAL_FOV);

  mLoadingScreen = new gui::GuiItem("file://../share/resources/gui/loading_screen.html");
  mSideBar       = new gui::GuiItem("file://../share/resources/gui/sidebar.html");
  mFooter        = new gui::GuiItem("file://../share/resources/gui/footer.html");
  mNotifications = new gui::GuiItem("file://../share/resources/gui/notifications.html");
  mLogo          = new gui::GuiItem("file://../share/resources/gui/logo.html");
  mStatistics    = new gui::GuiItem("file://../share/resources/gui/statistics.html");
  mTimeline      = new gui::GuiItem("file://../share/resources/gui/timeline.html");

  mLoadingScreen->setIsInteractive(false);

  // Add global gui items to mGlobalGuiArea if available, else use the mLocalGuiArea.
  if (mGlobalGuiArea) {
    mGlobalGuiArea->addItem(mLogo);
    mGlobalGuiArea->addItem(mNotifications);
    mGlobalGuiArea->addItem(mFooter);
    mGlobalGuiArea->addItem(mSideBar);
    mGlobalGuiArea->addItem(mTimeline);
    mGlobalGuiArea->addItem(mLoadingScreen);
  } else {
    mLocalGuiArea->addItem(mLogo);
    mLocalGuiArea->addItem(mNotifications);
    mLocalGuiArea->addItem(mFooter);
    mLocalGuiArea->addItem(mSideBar);
    mLocalGuiArea->addItem(mTimeline);
    mLocalGuiArea->addItem(mLoadingScreen);
  }

  mLocalGuiArea->addItem(mStatistics);

  mSideBar->setSizeX(500);
  mSideBar->setRelSizeY(1.f);
  mSideBar->setRelPositionY(1.f);
  mSideBar->setRelPositionX(0.f);
  mSideBar->setOffsetX(250);
  mSideBar->setRelOffsetY(-0.5f);
  mSideBar->setCursorChangeCallback([this](gui::Cursor c) { setCursor(c); });

  mFooter->setRelSizeX(1.f);
  mFooter->setSizeY(80);
  mFooter->setRelPositionX(0.5);
  mFooter->setRelPositionY(1.f);
  mFooter->setCursorChangeCallback([this](gui::Cursor c) { setCursor(c); });

  mTimeline->setRelSizeX(1.f);
  mTimeline->setSizeY(644);
  mTimeline->setRelPositionX(0.5);
  mTimeline->setRelPositionY(0);
  mTimeline->setOffsetY(322);
  mTimeline->setCursorChangeCallback([this](gui::Cursor c) { setCursor(c); });

  mNotifications->setSizeX(420);
  mNotifications->setSizeY(320);
  mNotifications->setRelPositionY(0.f);
  mNotifications->setRelPositionX(1.f);
  mNotifications->setOffsetX(-210);
  mNotifications->setOffsetY(250);
  mNotifications->setIsInteractive(false);

  mLogo->setSizeX(120);
  mLogo->setSizeY(100);
  mLogo->setRelPositionY(1.f);
  mLogo->setRelPositionX(1.f);
  mLogo->setOffsetX(-60);
  mLogo->setOffsetY(-50);
  mLogo->setIsInteractive(false);

  mStatistics->setSizeX(1200);
  mStatistics->setSizeY(300);
  mStatistics->setOffsetX(-600);
  mStatistics->setOffsetY(300);
  mStatistics->setRelPositionY(0.f);
  mStatistics->setRelPositionX(1.f);
  mStatistics->setIsInteractive(false);

  mLocalGuiOpenGLnode = pSG->NewOpenGLNode(mLocalGuiTransform, mLocalGuiArea);
  VistaOpenSGMaterialTools::SetSortKeyOnSubtree(
      mLocalGuiOpenGLnode, static_cast<int>(utils::DrawOrder::eGui));

  if (mGlobalGuiArea) {
    mGlobalGuiOpenGLnode = pSG->NewOpenGLNode(mGlobalGuiTransform, mGlobalGuiArea);
    mInputManager->registerSelectable(mGlobalGuiOpenGLnode);
    VistaOpenSGMaterialTools::SetSortKeyOnSubtree(
        mGlobalGuiTransform, static_cast<int>(utils::DrawOrder::eGui));
  }

  mInputManager->registerSelectable(mLocalGuiArea);

  mSideBar->waitForFinishedLoading();
  mFooter->waitForFinishedLoading();
  mTimeline->waitForFinishedLoading();
  mNotifications->waitForFinishedLoading();
  mLoadingScreen->waitForFinishedLoading();

  std::string version(GIT_RECENT_TAG);

  if (GIT_BRANCH != "") {
    version += " (" + GIT_BRANCH;
    if (GIT_COMMIT_HASH != "") {
      version += " @" + GIT_COMMIT_HASH;
    }
    version += ")";
  }

  mLoadingScreen->callJavascript("set_version", version);
  mLoadingScreen->callJavascript("set_loading", true);

  // Register callbacks for notifications area.
  mSideBar->registerCallback<std::string, std::string, std::string>("print_notification",
      ([this](std::string const& title, std::string const& content, std::string const& icon) {
        showNotification(title, content, icon);
      }));

  mTimeline->registerCallback<std::string, std::string, std::string>("print_notification",
      ([this](std::string const& title, std::string const& content, std::string const& icon) {
        showNotification(title, content, icon);
      }));

  // Register callbacks for sidebar area.

  mSideBar->registerCallback<bool>("set_enable_timer_queries",
      ([this](bool value) { mFrameTimings->pEnableMeasurements = value; }));

  mSideBar->registerCallback<bool>("set_enable_vsync", ([this](bool value) {
    GetVistaSystem()
        ->GetDisplayManager()
        ->GetWindows()
        .begin()
        ->second->GetWindowProperties()
        ->SetVSyncEnabled(value);
  }));

  mFrameTimings->pEnableMeasurements.onChange().connect(
      [this](bool enable) { mStatistics->setIsEnabled(enable); });

  // Set settings for the time Navigation
  mTimeline->callJavascript("setTimelineRange", settings->mMinDate, settings->mMaxDate);

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
  delete mFooter;
  delete mNotifications;
  delete mLogo;
  delete mGlobalGuiArea;
  delete mViewportUpdater;
  delete mTimeline;

  mInputManager->unregisterSelectable(mLocalGuiOpenGLnode);

  if (mGlobalGuiOpenGLnode) {
    mInputManager->unregisterSelectable(mGlobalGuiOpenGLnode);
  }

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

gui::GuiItem* GuiManager::getFooter() const {
  return mFooter;
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

void GuiManager::registerTool(std::shared_ptr<tools::Tool> const& tool) {
  mTools.push_back(tool);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void GuiManager::setLoadingScreenStatus(std::string const& sStatus) const {
  mLoadingScreen->callJavascript("set_status", sStatus);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void GuiManager::hideLoadingScreen() {
  if (mLoadingScreen) {
    if (mGlobalGuiArea) {
      mGlobalGuiArea->removeItem(mLoadingScreen);
    } else {
      mLocalGuiArea->removeItem(mLoadingScreen);
    }

    // All plugins finished loading -> init their custom components.
    mSideBar->callJavascript("init");

    mInputManager->pHoveredGuiNode = nullptr;

    delete mLoadingScreen;
    mLoadingScreen = nullptr;
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void setEnableStatistics(bool enable) {
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void GuiManager::showGui() {
  mGlobalGuiTransform->SetIsEnabled(true);
  mLocalGuiTransform->SetIsEnabled(true);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void GuiManager::hideGui() {
  mGlobalGuiTransform->SetIsEnabled(false);
  mLocalGuiTransform->SetIsEnabled(false);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void GuiManager::toggleGui() {
  mGlobalGuiTransform->SetIsEnabled(!mGlobalGuiTransform->GetIsEnabled());
  mLocalGuiTransform->SetIsEnabled(!mLocalGuiTransform->GetIsEnabled());
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void GuiManager::update() {
  for (auto it = mTools.begin(); it != mTools.end();) {
    if ((*it)->pShouldDelete.get()) {
      it = mTools.erase(it);
    } else {
      (*it)->update();
      ++it;
    }
  }

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

  // Update entire gui.
  gui::update();
}
void GuiManager::addPluginTabToSideBar(
    std::string const& name, std::string const& icon, std::string const& content) {
  mSideBar->callJavascript("addPluginTab", name, icon, content);
}

void GuiManager::addPluginTabToSideBarFromHTML(
    std::string const& name, std::string const& icon, std::string const& htmlFile) {
  std::string content = utils::filesystem::loadToString(htmlFile);
  addPluginTabToSideBar(name, icon, content);
}

void GuiManager::addSettingsSectionToSideBar(
    std::string const& name, std::string const& icon, std::string const& content) {
  mSideBar->callJavascript("addSettingsSection", name, icon, content);
}

void GuiManager::addSettingsSectionToSideBarFromHTML(
    std::string const& name, std::string const& icon, std::string const& htmlFile) {
  std::string content = utils::filesystem::loadToString(htmlFile);
  addSettingsSectionToSideBar(name, icon, content);
}

void GuiManager::addScriptToSideBar(std::string const& src) {
  mSideBar->executeJavascript(src);
}

void GuiManager::addScriptToSideBarFromJS(std::string const& jsFile) {
  std::string content = utils::filesystem::loadToString(jsFile);
  addScriptToSideBar(content);
}

void GuiManager::addEventToTimenavigationBar(std::string start, std::optional<std::string> end,
    std::string id, std::string content, std::optional<std::string> style, std::string description,
    std::string planet, std::string place) {
  mTimeline->callJavascript("addItem", start, end.value_or(""), id, content, style.value_or(""),
      description, planet, place);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cs::core
