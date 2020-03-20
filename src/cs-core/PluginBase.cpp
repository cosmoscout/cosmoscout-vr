////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "PluginBase.hpp"

namespace cs::core {

////////////////////////////////////////////////////////////////////////////////////////////////////

void PluginBase::setAPI(std::shared_ptr<Settings> const& settings,
    std::shared_ptr<SolarSystem> const& solarSystem, std::shared_ptr<GuiManager> const& guiManager,
    std::shared_ptr<InputManager> const& inputManager, VistaSceneGraph* sceneGraph,
    std::shared_ptr<GraphicsEngine> const&      graphicsEngine,
    std::shared_ptr<utils::FrameTimings> const& frameTimings,
    std::shared_ptr<TimeControl> const&         timeControl) {

  mAllSettings    = settings;
  mSolarSystem    = solarSystem;
  mSceneGraph     = sceneGraph;
  mGuiManager     = guiManager;
  mGraphicsEngine = graphicsEngine;
  mInputManager   = inputManager;
  mFrameTimings   = frameTimings;
  mTimeControl    = timeControl;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cs::core
