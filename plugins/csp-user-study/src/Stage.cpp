////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "Stage.hpp"
#include "logger.hpp"

#include "../../../src/cs-core/GuiManager.hpp"
#include "../../../src/cs-scene/CelestialAnchorNode.hpp"

#include <VistaKernel/GraphicsManager/VistaOpenGLNode.h>
#include <VistaKernel/GraphicsManager/VistaSceneGraph.h>
#include <VistaKernel/VistaSystem.h>
#include <VistaKernelOpenSGExt/VistaOpenSGMaterialTools.h>

namespace csp::userstudy {

Stage::Stage(Plugin::Settings::StageType            type,
    std::shared_ptr<cs::scene::CelestialAnchorNode> anchor, float scale)
    : mType(type)
    , mAnchor(std::move(anchor))
    , mScale(scale) {

  // Create WorldSpaceGuiArea for the element
  mGuiArea = std::make_unique<cs::gui::WorldSpaceGuiArea>(720, 720);
  mGuiArea->setUseLinearDepthBuffer(true);

  // Create Transform node to attach and scale the gui element
  VistaSceneGraph* pSG = GetVistaSystem()->GetGraphicsManager()->GetSceneGraph();
  mTransform.reset(pSG->NewTransformNode(mAnchor.get()));
  mTransform->Scale(mScale * static_cast<float>(mGuiArea->getWidth()),
      mScale * static_cast<float>(mGuiArea->getHeight()), 1.0F);

  // Attach OpenGLNode to Transform node containing WorldSpaceGuiArea
  mGuiNode.reset(pSG->NewOpenGLNode(mTransform.get(), mGuiArea.get()));
  VistaOpenSGMaterialTools::SetSortKeyOnSubtree(
      mGuiNode.get(), static_cast<int>(cs::utils::DrawOrder::eTransparentItems));

  // Add GuiItem to WorldSpaceGuiArea
  mGuiItem =
      std::make_unique<cs::gui::GuiItem>("file://../share/resources/gui/user-study-stage.html");
  mGuiArea->addItem(mGuiItem.get());
  mGuiItem->waitForFinishedLoading();
  // TODO: change test html code
  std::string testHTML = R"(
            <div style="width: 100%; height: 100%; background: lime;"></div>
        )";
  mGuiItem->callJavascript("setContent", testHTML);
}

} // namespace csp::userstudy