////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "OverlayRender.hpp"

#include <VistaKernel/VistaSystem.h>
#include <VistaKernel/GraphicsManager/VistaGraphicsManager.h>
#include <VistaKernel/GraphicsManager/VistaOpenGLNode.h>
#include <VistaKernel/GraphicsManager/VistaSceneGraph.h>
#include <VistaKernelOpenSGExt/VistaOpenSGMaterialTools.h>

#include "../../../src/cs-core/SolarSystem.hpp"

namespace csp::visualquery {

////////////////////////////////////////////////////////////////////////////////////////////////////

const std::string OverlayRender::sName = "OverlayRender";

////////////////////////////////////////////////////////////////////////////////////////////////////

std::string OverlayRender::sSource() {
  return cs::utils::filesystem::loadToString(
      "../share/resources/nodes/csp-visual-query/OverlayRender.js");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::unique_ptr<OverlayRender> OverlayRender::sCreate(
    std::shared_ptr<cs::core::SolarSystem> solarSystem,
    std::shared_ptr<cs::core::Settings>    settings) {
  return std::make_unique<OverlayRender>(std::move(solarSystem), std::move(settings));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::string const& OverlayRender::getName() const {
  return sName;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

OverlayRender::OverlayRender(std::shared_ptr<cs::core::SolarSystem> solarSystem,
    std::shared_ptr<cs::core::Settings>                             settings)
    : mSolarSystem(std::move(solarSystem))
    , mSettings(std::move(settings)) {
  mRenderer = std::make_unique<Renderer>("Earth", mSolarSystem, mSettings);

  // Add to scenegraph.
  VistaSceneGraph* pSG = GetVistaSystem()->GetGraphicsManager()->GetSceneGraph();
  mGLNode.reset(pSG->NewOpenGLNode(pSG->GetRoot(), mRenderer.get()));
  VistaOpenSGMaterialTools::SetSortKeyOnSubtree(
      mGLNode.get(), static_cast<int>(cs::utils::DrawOrder::ePlanets) + 10);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

OverlayRender::~OverlayRender() = default;

////////////////////////////////////////////////////////////////////////////////////////////////////

void OverlayRender::process() {
  auto input = readInput<std::shared_ptr<Image2D>>("Image2D", nullptr);
  mRenderer->setData(input);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::visualquery
