////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "DeletableMark.hpp"

#include "../../cs-core/GuiManager.hpp"
#include "../../cs-gui/GuiItem.hpp"
#include "../../cs-gui/WorldSpaceGuiArea.hpp"
#include "../../cs-scene/CelestialAnchorNode.hpp"
#include "../InputManager.hpp"
#include "../SolarSystem.hpp"

#include <VistaKernel/GraphicsManager/VistaOpenGLNode.h>
#include <VistaKernel/GraphicsManager/VistaSceneGraph.h>
#include <VistaKernel/VistaSystem.h>
#include <VistaKernelOpenSGExt/VistaOpenSGMaterialTools.h>

namespace cs::core::tools {

////////////////////////////////////////////////////////////////////////////////////////////////////

DeletableMark::DeletableMark(std::shared_ptr<InputManager> const& pInputManager,
    std::shared_ptr<SolarSystem> const& pSolarSystem, std::shared_ptr<Settings> const& settings,
    std::shared_ptr<TimeControl> const& pTimeControl, std::string const& sCenter,
    std::string const& sFrame)
    : Mark(pInputManager, pSolarSystem, settings, pTimeControl, sCenter, sFrame)
    , mGuiArea(new cs::gui::WorldSpaceGuiArea(65, 75))
    , mGuiItem(new cs::gui::GuiItem("file://../share/resources/gui/deletable_mark.html")) {

  initData();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

DeletableMark::~DeletableMark() {
  if (mGuiNode) {
    mGuiItem->unregisterCallback("deleteMe");
    mInputManager->unregisterSelectable(mGuiNode);
    mGuiArea->removeItem(mGuiItem.get());

    delete mGuiNode;
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void DeletableMark::initData() {
  auto* pSG = GetVistaSystem()->GetGraphicsManager()->GetSceneGraph();

  auto* pGuiTransform = pSG->NewTransformNode(mAnchor.get());

  pGuiTransform->Translate(0.F, 0.75F, 0.F);

  float const scale = 0.0005F;
  pGuiTransform->Scale(scale * static_cast<float>(mGuiArea->getWidth()),
      scale * static_cast<float>(mGuiArea->getHeight()), 1.F);

  pGuiTransform->Rotate(VistaAxisAndAngle(VistaVector3D(0.0, 1.0, 0.0), -glm::pi<float>() / 2.F));
  mGuiArea->addItem(mGuiItem.get());
  mGuiArea->setUseLinearDepthBuffer(true);

  mGuiItem->setCursorChangeCallback([](cs::gui::Cursor c) { cs::core::GuiManager::setCursor(c); });
  mGuiItem->setCanScroll(false);

  mGuiNode = pSG->NewOpenGLNode(pGuiTransform, mGuiArea.get());
  mInputManager->registerSelectable(mGuiNode);

  VistaOpenSGMaterialTools::SetSortKeyOnSubtree(
      pGuiTransform, static_cast<int>(cs::utils::DrawOrder::eTransparentItems));

  mGuiItem->registerCallback("deleteMe", "Call this to remove the tool.",
      std::function([this]() { pShouldDelete = true; }));

  mSelfSelectedConnection =
      pSelected.connect([this](bool val) { mGuiItem->callJavascript("setMinimized", !val); });
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cs::core::tools
