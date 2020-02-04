////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CS_APPLICATION_HPP
#define CS_APPLICATION_HPP

#include <VistaKernel/VistaFrameLoop.h>
#include <map>
#include <memory>

#ifdef __linux__
#include "dlfcn.h"
#define COSMOSCOUT_LIBTYPE void*
#else
#include <windows.h>
#define COSMOSCOUT_LIBTYPE HINSTANCE
#endif

class IVistaClusterDataSync;

namespace cs::core {
class PluginBase;
class Settings;
class GuiManager;
class InputManager;
class GraphicsEngine;
class TimeControl;
class SolarSystem;
class DragNavigation;
} // namespace cs::core

namespace cs::utils {
class FrameTimings;
class Downloader;
} // namespace cs::utils

/// This is the core class of CosmoScout VR. The application and all plugins are initialized and
/// updated in the following sequence:
///   1. Application::Application().
///      - This only initializes the curl library.
///   2. Application::Init()
///      - Creates InputManager, FrameTimings, GraphicsEngine, GuiManager, TimeControl and
///        SolarSystem
///      - Application::connectSlots()
///      - Application::registerGuiCallbacks()
///      - Opens the plugin libraries and calls the "create" method of each individual plugin. This
///        usually means that the constructor of the Plugin class is called. See
///        cs::core::PluginBase for more details.
///   3. Application::FrameUpdate() is called once a frame
///      - In the first few frames only ViSTA classes are updated and the GuiManager. This ensures
///        that the loading screen is shown.
///      - After a few seconds, the data download in a background thread is started. Until
///        everything is downloaded, the progress is shown on the loading screen.
///      - SolarSystem::init() is called once the data download has finished.
///      - For each plugin:
///        - Show plugin name on the loading screen and wait some frames to make sure that the text
///          is updated.
///        - PluginBase::setAPI()
///        - PluginBase::init()
///      - When the last plugin finished loading, the loading screen is removed and the observer is
///        animated to its initial position in space.
///      - If all plugins are loaded:
///        - InputManager::update()
///        - TimeControl::update()
///        - PluginBase::update() for each plugin
///        - SolarSystem::update()
///        - GraphicsEngine::Update()
///      - GuiManager::update()
///   4. Application::~Application()
///      - For each plugin:
///        - PluginBase::deInit()
///        - Call the "destroy" method of the plugin. This usually means that the destructor of the
///          Plugin class is called. See cs::core::PluginBase for more details.
///      - SolarSystem::deinit()
///      - Cleanup curl
class Application : public VistaFrameLoop {
 public:
  /// This does only inititlize curl.
  explicit Application(cs::core::Settings const& settings);
  ~Application() override;

  /// Initializes the Application. Should only be called by ViSTA.
  bool Init(VistaSystem* pVistaSystem) override;

  /// De-Initializes the Application. Should only be called by ViSTA.
  void Quit() override;

  /// Called every frame by ViSTA. The whole application logic is executed here.
  void FrameUpdate() override;

  static void testLoadAllPlugins();

 private:
  struct Plugin {
    COSMOSCOUT_LIBTYPE    mHandle;
    cs::core::PluginBase* mPlugin = nullptr;
  };

  /// This connects several parts of CosmoScout VR to each other. For example, when the InputManager
  /// calculates a new intersection between the mouse-ray and the currently active planet, the
  /// coordinates of this intersection are shown in the user interface. Since the InputManager has
  /// no access to the GUI, this connection is established in this method.
  void connectSlots();

  /// There are several C++ callbacks available in the JavaScript code of the user interface. In
  /// this method those callbacks are set up. Here are all registered callbacks:
  ///  - print_notification
  ///  - set_celestial_body
  ///  - set_date
  ///  - set_time
  ///  - set_enable_cascades_debug
  ///  - set_ambient_light
  ///  - set_enable_lighting
  ///  - set_enable_shadow_freeze
  ///  - set_enable_shadows
  ///  - set_enable_timer_queries
  ///  - set_enable_vsync
  ///  - set_lighting_quality
  ///  - set_shadowmap_bias
  ///  - set_shadowmap_cascades
  ///  - set_shadowmap_extension
  ///  - set_shadowmap_range
  ///  - set_shadowmap_resolution
  ///  - set_shadowmap_split_distribution
  ///  - set_terrain_height
  ///  - set_widget_scale
  ///  - print_notification
  ///  - reset_time
  ///  - add_hours
  ///  - add_hours_without_animation
  ///  - set_time_speed
  ///  - navigate_north_up
  ///  - navigate_fix_horizon
  ///  - navigate_to_surface
  ///  - navigate_to_orbit
  void registerGuiCallbacks();
  void unregisterGuiCallbacks();

  std::shared_ptr<const cs::core::Settings> mSettings;
  std::shared_ptr<cs::core::InputManager>   mInputManager;
  std::shared_ptr<cs::core::GraphicsEngine> mGraphicsEngine;
  std::shared_ptr<cs::core::GuiManager>     mGuiManager;
  std::shared_ptr<cs::core::TimeControl>    mTimeControl;
  std::shared_ptr<cs::core::SolarSystem>    mSolarSystem;
  std::unique_ptr<cs::core::DragNavigation> mDragNavigation;
  std::shared_ptr<cs::utils::FrameTimings>  mFrameTimings;
  std::map<std::string, Plugin>             mPlugins;
  std::unique_ptr<cs::utils::Downloader>    mDownloader;
  std::unique_ptr<IVistaClusterDataSync>    mSceneSync;

  bool mDownloadedData            = false;
  bool mLoadedAllPlugins          = false;
  int  mStartPluginLoadingAtFrame = 0;
  int  mHideLoadingScreenAtFrame  = 0;
};

#endif // CS_APPLICATION_HPP
