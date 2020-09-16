////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "FlagTool.hpp"

#include "logger.hpp"

#include "../../../src/cs-core/GuiManager.hpp"
#include "../../../src/cs-core/InputManager.hpp"
#include "../../../src/cs-core/Settings.hpp"
#include "../../../src/cs-core/SolarSystem.hpp"
#include "../../../src/cs-core/TimeControl.hpp"
#include "../../../src/cs-scene/CelestialAnchorNode.hpp"
#include "../../../src/cs-utils/convert.hpp"

#include <VistaKernel/GraphicsManager/VistaOpenGLNode.h>
#include <VistaKernel/GraphicsManager/VistaSceneGraph.h>
#include <VistaKernel/VistaSystem.h>
#include <VistaKernelOpenSGExt/VistaOpenSGMaterialTools.h>

namespace csp::measurementtools {

////////////////////////////////////////////////////////////////////////////////////////////////////

FlagTool::FlagTool(std::shared_ptr<cs::core::InputManager> const& pInputManager,
    std::shared_ptr<cs::core::SolarSystem> const&                 pSolarSystem,
    std::shared_ptr<cs::core::Settings> const&                    settings,
    std::shared_ptr<cs::core::TimeControl> const& pTimeControl, std::string const& sCenter,
    std::string const& sFrame)
    : Mark(pInputManager, pSolarSystem, settings, pTimeControl, sCenter, sFrame)
    , mGuiArea(std::make_unique<cs::gui::WorldSpaceGuiArea>(600, 400))
    , mGuiItem(std::make_unique<cs::gui::GuiItem>(
          "file://{toolZoom}../share/resources/gui/flag.html", true)) {
  auto* pSG = GetVistaSystem()->GetGraphicsManager()->GetSceneGraph();

  mGuiTransform.reset(pSG->NewTransformNode(mAnchor.get()));
  mGuiTransform->Translate(0.5F, 0.5F, 0.F);
  mGuiTransform->Scale(0.0005F * static_cast<float>(mGuiArea->getWidth()),
      0.0005F * static_cast<float>(mGuiArea->getHeight()), 1.F);
  mGuiTransform->Rotate(VistaAxisAndAngle(VistaVector3D(0.0, 1.0, 0.0), -glm::pi<float>() / 2.F));
  mGuiArea->addItem(mGuiItem.get());
  mGuiArea->setUseLinearDepthBuffer(true);

  mGuiNode.reset(pSG->NewOpenGLNode(mGuiTransform.get(), mGuiArea.get()));
  mInputManager->registerSelectable(mGuiNode.get());

  VistaOpenSGMaterialTools::SetSortKeyOnSubtree(
      mGuiNode.get(), static_cast<int>(cs::utils::DrawOrder::eTransparentItems));

  mGuiItem->setCanScroll(false);
  mGuiItem->waitForFinishedLoading();

  // We use a zoom factor of 2.0 in order to increae the DPI of our world space UIs.
  mGuiItem->setZoomFactor(2.0);

  mGuiItem->registerCallback("deleteMe", "Call this to delete the tool.",
      std::function([this]() { pShouldDelete = true; }));
  mGuiItem->setCursorChangeCallback([](cs::gui::Cursor c) { cs::core::GuiManager::setCursor(c); });

  // Update text.
  mTextConnection = pText.connectAndTouch(
      [this](std::string const& value) { mGuiItem->callJavascript("setText", value); });

  mGuiItem->registerCallback("onSetText",
      "This is called whenever the text input of the tool's name changes.",
      std::function(
          [this](std::string&& value) { pText.setWithEmitForAllButOne(value, mTextConnection); }));

  // Update position.
  pLngLat.connect([this](glm::dvec2 const& lngLat) {
    auto body = mSolarSystem->getBody(mAnchor->getCenterName());
    if (body) {
      double h = body->getHeight(lngLat);
      mGuiItem->callJavascript("setPosition", cs::utils::convert::toDegrees(lngLat.x),
          cs::utils::convert::toDegrees(lngLat.y), h);
    }
  });

  // Update minimized state.
  mDoubleClickConnection = mInputManager->sOnDoubleClick.connect([this]() {
    if (pHovered.get()) {
      pMinimized = !pMinimized.get();
    }
  });

  pMinimized.connect([this](bool val) { mGuiItem->callJavascript("setMinimized", val); });

  mGuiItem->registerCallback("minimizeMe", "Call this to minimize the flag.",
      std::function([this]() { pMinimized = true; }));
  mGuiItem->callJavascript("setActivePlanet", sCenter);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

FlagTool::~FlagTool() {
  auto* pSG = GetVistaSystem()->GetGraphicsManager()->GetSceneGraph();
  pSG->GetRoot()->DisconnectChild(mGuiTransform.get());

  mInputManager->sOnDoubleClick.disconnect(mDoubleClickConnection);
  mInputManager->unregisterSelectable(mGuiNode.get());
  mGuiItem->unregisterCallback("minimizeMe");
  mGuiItem->unregisterCallback("deleteMe");
  mGuiItem->unregisterCallback("onSetText");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void FlagTool::update() {
  // This seems to be the first time the tool is updated, so we have to store the distance to the
  // observer so that we can scale the tool later based on the observer's position.
  if (pScaleDistance.get() < 0) {
    try {
      pScaleDistance = mSolarSystem->getObserver().getAnchorScale() *
                       glm::length(mSolarSystem->getObserver().getRelativePosition(
                           mTimeControl->pSimulationTime.get(), *mAnchor));
    } catch (std::exception const& e) {
      // Getting the relative transformation may fail due to insufficient SPICE data.
      logger().warn("Failed to calculate scale distance of Flag Tool: {}", e.what());
    }
  }

  double simulationTime(mTimeControl->pSimulationTime.get());

  cs::core::SolarSystem::scaleRelativeToObserver(*mAnchor, mSolarSystem->getObserver(),
      simulationTime, pScaleDistance.get(), mSettings->mGraphics.pWorldUIScale.get());
  cs::core::SolarSystem::turnToObserver(
      *mAnchor, mSolarSystem->getObserver(), simulationTime, true);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::measurementtools
