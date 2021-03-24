////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CS_CORE_PLUGIN_BASE_HPP
#define CS_CORE_PLUGIN_BASE_HPP

#include "cs_core_export.hpp"

#include <memory>

#ifdef __linux__
#define EXPORT_FN extern "C" __attribute__((visibility("default")))
#else
#define EXPORT_FN extern "C" __declspec(dllexport)
#endif

class VistaSceneGraph;

namespace cs::core {
class GraphicsEngine;
class GuiManager;
class InputManager;
class TimeControl;
class SolarSystem;
class Settings;

/// The base interface for all plugins. This class serves as an entry point and gives the plugin
/// a hook into the update loop.
class CS_CORE_EXPORT PluginBase {
 public:
  /// The constructor is called when the plugin is opened. This should contain no or very little
  /// code. Your heavy initialization code should be executed in init().
  PluginBase() = default;

  PluginBase(PluginBase const& other) = delete;
  PluginBase(PluginBase&& other)      = delete;

  PluginBase& operator=(PluginBase const& other) = delete;
  PluginBase& operator=(PluginBase&& other) = delete;

  virtual ~PluginBase() = default;

  /// This initializes the program state for this plugin, so it can interact with it. It will be
  /// called by the core application before the plugin initializes. There is no need to call this
  /// function from the plugin.
  void setAPI(std::shared_ptr<Settings> settings, std::shared_ptr<SolarSystem> solarSystem,
      std::shared_ptr<GuiManager> guiManager, std::shared_ptr<InputManager> inputManager,
      VistaSceneGraph* sceneGraph, std::shared_ptr<GraphicsEngine> graphicsEngine,
      std::shared_ptr<TimeControl> timeControl);

  /// Override this function to initialize your plugin. It will be called directly after
  /// application startup and before the update loop starts.
  virtual void init(){};

  /// Override this function for cleaning up after yourself, when the plugin terminates. We don't
  /// want our app littered :)
  virtual void deInit(){};

  /// Override this function if you want to do something in every frame. See the Application class
  /// for more details on when this method is actually called.
  virtual void update(){};

 protected:
  std::shared_ptr<Settings>       mAllSettings;
  std::shared_ptr<SolarSystem>    mSolarSystem;
  VistaSceneGraph*                mSceneGraph{};
  std::shared_ptr<GuiManager>     mGuiManager;
  std::shared_ptr<GraphicsEngine> mGraphicsEngine;
  std::shared_ptr<InputManager>   mInputManager;
  std::shared_ptr<TimeControl>    mTimeControl;
};

} // namespace cs::core

#endif // CS_CORE_PLUGIN_BASE_HPP
