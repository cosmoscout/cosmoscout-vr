////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "DeletableMark.hpp"

#include "../../cs-gui/GuiItem.hpp"
#include "../../cs-gui/WorldSpaceGuiArea.hpp"
#include "../../cs-scene/CelestialAnchorNode.hpp"
#include "../GuiManager.hpp"
#include "../InputManager.hpp"
#include "../SolarSystem.hpp"

#include <VistaKernel/GraphicsManager/VistaOpenGLNode.h>
#include <VistaKernel/GraphicsManager/VistaSceneGraph.h>
#include <VistaKernel/VistaSystem.h>
#include <VistaKernelOpenSGExt/VistaOpenSGMaterialTools.h>

namespace cs::core::tools {

////////////////////////////////////////////////////////////////////////////////////////////////////

DeletableMark::DeletableMark(std::shared_ptr<InputManager> const& pInputManager,
    std::shared_ptr<SolarSystem> const&                           pSolarSystem,
    std::shared_ptr<GraphicsEngine> const&                        graphicsEngine,
    std::shared_ptr<GuiManager> const&                            pGuiManager,
    std::shared_ptr<TimeControl> const& pTimeControl, std::string const& sCenter,
    std::string const& sFrame)
    : Mark(pInputManager, pSolarSystem, graphicsEngine, pGuiManager, pTimeControl, sCenter, sFrame)
    , mGuiManager(pGuiManager)
    , mGuiArea(new cs::gui::WorldSpaceGuiArea(80, 90))
    , mGuiItem(new cs::gui::GuiItem("file://../share/resources/gui/deletable_mark.html")) {

  initData();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

DeletableMark::DeletableMark(DeletableMark const& other)
    : Mark(other)
    , mGuiManager(other.mGuiManager)
    , mGuiArea(new cs::gui::WorldSpaceGuiArea(100, 100))
    , mGuiItem(new cs::gui::GuiItem("file://../share/resources/gui/deletable_mark.html")) {

  initData();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

DeletableMark::~DeletableMark() {
  if (mGuiNode) {
    mGuiItem->unregisterCallback("delete_me");
    mInputManager->unregisterSelectable(mGuiNode);
    mGuiArea->removeItem(mGuiItem.get());

    delete mGuiNode;
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void DeletableMark::initData() {
  auto pSG = GetVistaSystem()->GetGraphicsManager()->GetSceneGraph();

  auto pGuiTransform = pSG->NewTransformNode(mAnchor.get());
  pGuiTransform->Translate(0.f, 0.4f, 0.f);
  pGuiTransform->Scale(0.001f * mGuiArea->getWidth(), 0.001f * mGuiArea->getHeight(), 1.f);
  pGuiTransform->Rotate(VistaAxisAndAngle(VistaVector3D(0.0, 1.0, 0.0), -glm::pi<float>() / 2.f));
  mGuiArea->addItem(mGuiItem.get());
  mGuiArea->setUseLinearDepthBuffer(true);

  mGuiItem->setCursorChangeCallback([this](cs::gui::Cursor c) { mGuiManager->setCursor(c); });

  mGuiNode = pSG->NewOpenGLNode(pGuiTransform, mGuiArea.get());
  mInputManager->registerSelectable(mGuiNode);

  VistaOpenSGMaterialTools::SetSortKeyOnSubtree(
      pGuiTransform, static_cast<int>(cs::utils::DrawOrder::eTransparentItems));

  mGuiItem->registerCallback("delete_me", [this]() { pShouldDelete = true; });

  mSelfSelectedConnection = pSelected.onChange().connect(
      [this](bool val) { mGuiItem->callJavascript("set_minimized", !val); });
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cs::core::tools
