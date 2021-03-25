////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "PluginBase.hpp"

namespace cs::core {

////////////////////////////////////////////////////////////////////////////////////////////////////

void PluginBase::setAPI(std::shared_ptr<Settings> settings,
    std::shared_ptr<SolarSystem> solarSystem, std::shared_ptr<GuiManager> guiManager,
    std::shared_ptr<InputManager> inputManager, VistaSceneGraph* sceneGraph,
    std::shared_ptr<GraphicsEngine> graphicsEngine, std::shared_ptr<TimeControl> timeControl) {

  mAllSettings    = std::move(settings);
  mSolarSystem    = std::move(solarSystem);
  mSceneGraph     = sceneGraph;
  mGuiManager     = std::move(guiManager);
  mGraphicsEngine = std::move(graphicsEngine);
  mInputManager   = std::move(inputManager);
  mTimeControl    = std::move(timeControl);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cs::core
