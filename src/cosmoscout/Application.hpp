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
} // namespace cs::utils

/// This is the core class of CosmoScout VR. It is responsible for setting all ViSTA modules up,
/// loading the plugins and running the frame loop.
class Application : public VistaFrameLoop {
 public:
  explicit Application(cs::core::Settings const& settings);
  ~Application() override;

  /// Initializes the Application. Should only be called by ViSTA.
  bool Init(VistaSystem* pVistaSystem) override;

  /// Called every frame. The whole application logic is executed here.
  void FrameUpdate() override;

 private:
  struct Plugin {
    COSMOSCOUT_LIBTYPE    mHandle;
    cs::core::PluginBase* mPlugin = nullptr;
  };

  void connectSlots();

  /// This scales the cs::scene::CelestialObserver of the solar system to move the
  /// closest body to a small world space distance. This distance depends on his or her *real*
  /// distance in outer space to the respective body.
  /// In order for the scientists to be able to interact with their environment, the next virtual
  /// celestial body must never be more than an arm’s length away. If the Solar System were always
  /// represented on a 1:1 scale, the virtual planetary surface would be too far away to work
  /// effectively with the simulation.
  /// As objects will be quite close to the observer in world space if the user is far away in
  /// *real* space, this also reduces the far clip distance in order to increase depth accuracy
  /// for objects close to the observer. This method also manages the SPICE frame changes when the
  /// observer moves from body to body.
  void updateSceneScale();

  std::shared_ptr<const cs::core::Settings> mSettings;
  std::shared_ptr<cs::core::InputManager>   mInputManager;
  std::shared_ptr<cs::core::GraphicsEngine> mGraphicsEngine;
  std::shared_ptr<cs::core::GuiManager>     mGuiManager;
  std::shared_ptr<cs::core::TimeControl>    mTimeControl;
  std::shared_ptr<cs::core::SolarSystem>    mSolarSystem;
  std::shared_ptr<cs::core::DragNavigation> mDragNavigation;
  std::shared_ptr<cs::utils::FrameTimings>  mFrameTimings;
  std::map<std::string, Plugin>             mPlugins;
  bool                                      mLoadedAllPlugins = false;

  std::unique_ptr<IVistaClusterDataSync> mSceneSync;
};

#endif // CS_APPLICATION_HPP
