////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef VIRTUAL_PLANET_APPLICATION_HPP
#define VIRTUAL_PLANET_APPLICATION_HPP

#include "../cs-core/PluginBase.hpp"
#include "../cs-core/Settings.hpp"
#include "../cs-graphics/MouseRay.hpp"
#include <VistaKernel/GraphicsManager/VistaOpenGLNode.h>
#include <VistaKernel/VistaFrameLoop.h>
#include <VistaKernel/VistaSystem.h>

#ifdef __linux__
#include "dlfcn.h"
#define COSMOSCOUT_LIBTYPE void*
#else
#include <windows.h>
#define COSMOSCOUT_LIBTYPE HINSTANCE
#endif

namespace cs::core {
class GuiManager;
class InputManager;
class GraphicsEngine;
class TimeControl;
class SolarSystem;
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

  std::shared_ptr<const cs::core::Settings> mSettings;
  std::shared_ptr<cs::core::InputManager>   mInputManager;
  std::shared_ptr<cs::core::GraphicsEngine> mGraphicsEngine;
  std::shared_ptr<cs::core::GuiManager>     mGuiManager;
  std::shared_ptr<cs::core::TimeControl>    mTimeControl;
  std::shared_ptr<cs::core::SolarSystem>    mSolarSystem;
  std::shared_ptr<cs::utils::FrameTimings>  mFrameTimings;
  std::map<std::string, Plugin>             mPlugins;
  bool                                      mLoadedAllPlugins = false;

  void registerSolarSystemCallbacks();
  void registerTimenavigationBarCallbacks();
  void registerSideBarCallbacks();
  void registerCalendarCallbacks();
};

#endif // VIRTUAL_PLANET_APPLICATION_HPP
