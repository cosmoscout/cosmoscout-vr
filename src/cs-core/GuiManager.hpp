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

/// The GuiManager is the central access point to the applications user interface. It is passed to
/// all plugins, so they get info about the GUI or modify it.
class CS_CORE_EXPORT GuiManager {
 public:
  GuiManager(std::shared_ptr<const Settings> const& settings,
      std::shared_ptr<InputManager> const&          pInputManager,
      std::shared_ptr<utils::FrameTimings> const&   pFrameTimings);
  virtual ~GuiManager();

  /// Set the cursor icon.
  void setCursor(gui::Cursor cursor);

  /// Shows a notification in the top right corner.
  ///
  /// @param sIcon         The name of the material theme icon the notification should display.
  /// @param sFlyToOnClick The name of a location to fly to when clicked.
  void showNotification(std::string const& sTitle, std::string const& sText,
      std::string const& sIcon, std::string const& sFlyToOnClick = "") const;

  /// Sets the status text of the loading screen.
  void setLoadingScreenStatus(std::string const& sStatus) const;

  /// Hides the loading screen.
  void hideLoadingScreen();

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

  /// Adds an initialization script to the sidebar. This should be called after all tabs and
  /// sections have been added.
  ///
  /// @param src The javascript source code.
  void addScriptToSideBar(std::string const& src);

  /// Adds an initialization script to the sidebar. This should be called after all tabs and
  /// sections have been added.
  ///
  /// @param jsFile The javascript file that contains the source code.
  void addScriptToSideBarFromJS(std::string const& jsFile);

  /// Adds an event item to the timenavigation
  ///
  /// @param start The start date of the event.
  /// @param end The optional end date of the event.
  /// @param id The id of the event.
  /// @param content The name or content of the event.
  /// @param style The optional css of the event.
  /// @param description The description of the event.
  /// @param planet Planet the event is happening on.
  /// @parama place The location on the planet.
  void addEventToTimenavigationBar(std::string start, std::optional<std::string> end,
      std::string id, std::string content, std::optional<std::string> style,
      std::string description, std::string planet, std::string place);

  /// Returns the side bar GuiItem. The side bar is located at the left side of the screen.
  gui::GuiItem* getSideBar() const;

  /// Returns the header bar GuiItem. The header bar is at the top of the screen.
  gui::GuiItem* getFooter() const;

  /// Returns the time navigation bar GuiItem. The time navigation bar bar is at the bottom of the
  /// screen.
  gui::GuiItem* getTimeline() const;

  /// Returns the statistics GuiItem. The statistics are at the right of the screen, when enabled.
  gui::GuiItem* getStatistics() const;

  /// Returns the logo GuiItem. The logo is at the bottom right of the screen.
  gui::GuiItem* getLogo() const;

  void registerTool(std::shared_ptr<tools::Tool> const& tool);

  /// Toggles the statistics.
  void setEnableStatistics(bool enable);

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
  gui::GuiItem* mSideBar       = nullptr;
  gui::GuiItem* mFooter        = nullptr;
  gui::GuiItem* mNotifications = nullptr;
  gui::GuiItem* mLogo          = nullptr;
  gui::GuiItem* mStatistics    = nullptr;
  gui::GuiItem* mTimeline      = nullptr;

  VistaTransformNode* mGlobalGuiTransform  = nullptr;
  VistaTransformNode* mLocalGuiTransform   = nullptr;
  VistaOpenGLNode*    mLocalGuiOpenGLnode  = nullptr;
  VistaOpenGLNode*    mGlobalGuiOpenGLnode = nullptr;

  std::list<std::shared_ptr<tools::Tool>> mTools;

  int   mGuiWidth = 1920, mGuiHeight = 1080;
  float mGuiScaleX = 1.6f, mGuiScaleY = 0.9f;
  float mGuiPosX = 0.f, mGuiPosY = 0.f, mGuiPosZ = 0.f;
  float mGuiRotX = 0.f, mGuiRotY = 0.f, mGuiRotZ = 0.f;
};

} // namespace cs::core

#endif // CS_CORE_GUI_MANAGER_HPP
