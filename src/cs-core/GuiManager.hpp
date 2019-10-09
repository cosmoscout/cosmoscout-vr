////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CS_CORE_GUI_MANAGER_HPP
#define CS_CORE_GUI_MANAGER_HPP

#include "cs_core_export.hpp"

#include "../cs-gui/GuiItem.hpp"
#include "../cs-gui/ScreenSpaceGuiArea.hpp"
#include "../cs-gui/WorldSpaceGuiArea.hpp"
#include "../cs-gui/gui.hpp"
#include "../cs-gui/types.hpp"

#include "../cs-utils/FrameTimings.hpp"

#include <memory>
#include <string>

class VistaOpenGLNode;
class VistaViewportResizeToProjectionAdapter;
class VistaTransformNode;

namespace cs::utils {
class FrameTimings;
} // namespace cs::utils

namespace cs::core {

namespace tools {
class Tool;
}

class Settings;
class InputManager;

/// The GuiManager is the central access point to the application's user interface.
/// The user interface of CosmoScout VR consists of several webpages (GuiItems) which are rendered
/// with the Chromium Embedded Framework.
///
/// The GuiItems are either drawn in screen-space or - if the settings key "gui": {...} is specified
/// - in world-space. The key differences are:
/// Screen-Space:
///  * The UI automatically resizes when the window is resized
///  * When running in a clustered setup, each display will show an individual copy of the same
///    item. This is for example useful for the statistics GuiItem which is in all cases shown in
///    screen-space.
/// World-Space:
///  * The UI is drawn in a fixed resolution which is specified in the "gui": {...} settings key.
///  * When running in a clustered setup, the UI will be displayed across multiple displays.
///
/// There are several GuiItems involved: e.g. the header-bar, the side-bar and the
/// notifications area. There are methods for getting access to these GuiItems - for example, these
/// can be used to register callbacks which will be executed when a button is pressed in the UI.
/// Plugins can add content to the sidebar. This is done with the methods addPluginTabToSideBar(),
/// addSettingsSectionToSideBar() and addScriptToSideBar().
///
/// This class should only be instantiated once - this is done by the Application class and this
/// instance is then passed to all plugins.
class CS_CORE_EXPORT GuiManager {
 public:
  GuiManager(std::shared_ptr<const Settings> const& settings,
      std::shared_ptr<InputManager> const&          pInputManager,
      std::shared_ptr<utils::FrameTimings> const&   pFrameTimings);
  virtual ~GuiManager();

  /// Set the cursor icon. This is usually used in the following way:
  /// guiItem->setCursorChangeCallback(
  ///    [guiManager](cs::gui::Cursor c) { guiManager->setCursor(c); });
  void setCursor(gui::Cursor cursor);

  /// Shows a notification in the top right corner.
  ///
  /// @param sTitle        The first line of the notification.
  /// @param sText         The second line of the notification.
  /// @param sIcon         The name of the material theme icon the notification should display.
  /// @param sFlyToOnClick The name of a location to fly to when clicked.
  void showNotification(std::string const& sTitle, std::string const& sText,
      std::string const& sIcon, std::string const& sFlyToOnClick = "") const;

  /// Adds a new tab to the side bar.
  ///
  /// @param name     The nam/title of the tab.
  /// @param icon     The name of the Material icon.
  /// @param content  The HTML that describes the tabs contents.
  void addPluginTabToSideBar(
      std::string const& name, std::string const& icon, std::string const& content);

  /// Adds a new tab to the side bar.
  ///
  /// @param name      The nam/title of the tab.
  /// @param icon      The name of the Material icon.
  /// @param htmlFile  The HTML file that describes the tabs contents.
  void addPluginTabToSideBarFromHTML(
      std::string const& name, std::string const& icon, std::string const& htmlFile);

  /// Adds a new section to the settings tab.
  ///
  /// @param name     The name/title of the section.
  /// @param content  The HTML that describes the sections contents.
  void addSettingsSectionToSideBar(
      std::string const& name, std::string const& icon, std::string const& content);

  /// Adds a new section to the settings tab.
  ///
  /// @param name      The name/title of the section.
  /// @param htmlFile  The HTML file that describes the sections contents.
  void addSettingsSectionToSideBarFromHTML(
      std::string const& name, std::string const& icon, std::string const& htmlFile);

  /// This can be used to initialize the DOM elements added to the sidebar with the methods above.
  /// This is identical to getSideBar()->executeJavascript(src);
  ///
  /// @param src The javascript source code.
  void addScriptToSideBar(std::string const& src);

  /// This can be used to initialize the DOM elements added to the sidebar with the methods above.
  ///
  /// @param jsFile The javascript file that contains the source code.
  void addScriptToSideBarFromJS(std::string const& jsFile);

  /// Returns the side bar GuiItem. The side bar is located at the left side of the screen.
  gui::GuiItem* getSideBar() const;

  /// Returns the calender GuiItem. The calender is at the top of the screen, when clicked on the
  /// current date display.
  gui::GuiItem* getCalendar() const;

  /// Returns the header bar GuiItem. The header bar is at the top of the screen.
  gui::GuiItem* getHeaderBar() const;

  /// Returns the statistics GuiItem. The statistics are at the right of the screen, when enabled.
  gui::GuiItem* getStatistics() const;

  /// Returns the logo GuiItem. The logo is at the bottom right of the screen.
  gui::GuiItem* getLogo() const;

  /// Shows or hides the loading screen.
  void enableLoadingScreen(bool enable);

  /// Sets the status text on the loading screen. This is only useful during application start-up,
  /// as the loading screen will be hidden thereafter.
  void setLoadingScreenStatus(std::string const& sStatus) const;

  /// Sets the progress bar state.
  void setLoadingScreenProgress(float percent, bool animate) const;

  /// If you instantiate a cs::tools::Tool, you should register it here. The GuiManager will then
  /// call the update() of the tool each frame. The tool will be released automatically when its
  /// pShouldDelete property is set to true.
  void registerTool(std::shared_ptr<tools::Tool> const& tool);

  /// Hides or shows the entire user interface. This is bound to the ESC-key.
  void showGui();
  void hideGui();
  void toggleGui();

  /// This is called once a frame from the Application.
  void update();

 private:
  std::shared_ptr<InputManager>        mInputManager;
  std::shared_ptr<utils::FrameTimings> mFrameTimings;

  VistaViewportResizeToProjectionAdapter* mViewportUpdater = nullptr;
  gui::WorldSpaceGuiArea*                 mGlobalGuiArea   = nullptr;
  gui::ScreenSpaceGuiArea*                mLocalGuiArea    = nullptr;

  gui::GuiItem* mLoadingScreen = nullptr;
  gui::GuiItem* mCalendar      = nullptr;
  gui::GuiItem* mSideBar       = nullptr;
  gui::GuiItem* mHeaderBar     = nullptr;
  gui::GuiItem* mNotifications = nullptr;
  gui::GuiItem* mLogo          = nullptr;
  gui::GuiItem* mStatistics    = nullptr;

  // The global GUI is drawn in world-space.
  VistaTransformNode* mGlobalGuiTransform  = nullptr;
  VistaOpenGLNode*    mGlobalGuiOpenGLnode = nullptr;

  // The local GUI is drawn in screen-space.
  VistaTransformNode* mLocalGuiTransform  = nullptr;
  VistaOpenGLNode*    mLocalGuiOpenGLnode = nullptr;

  std::list<std::shared_ptr<tools::Tool>> mTools;

  int   mGuiWidth = 1920, mGuiHeight = 1080;
  float mGuiScaleX = 1.6f, mGuiScaleY = 0.9f;
  float mGuiPosX = 0.f, mGuiPosY = 0.f, mGuiPosZ = 0.f;
  float mGuiRotX = 0.f, mGuiRotY = 0.f, mGuiRotZ = 0.f;
};

} // namespace cs::core

#endif // CS_CORE_GUI_MANAGER_HPP
