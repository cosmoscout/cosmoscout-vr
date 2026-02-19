////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "Arrows.hpp"

#include "../../../src/cs-core/SolarSystem.hpp"
#include "../../../src/cs-utils/FrameStats.hpp"
#include "logger.hpp"

#include <VistaKernel/GraphicsManager/VistaGraphicsManager.h>
#include <VistaKernel/GraphicsManager/VistaSceneGraph.h>
#include <VistaKernel/GraphicsManager/VistaTransformNode.h>
#include <VistaKernel/VistaSystem.h>
#include <VistaKernelOpenSGExt/VistaOpenSGMaterialTools.h>

namespace csp::coordinatearrows {

////////////////////////////////////////////////////////////////////////////////////////////////////

Arrows::Arrows(/*std::shared_ptr<Plugin::Settings> pluginSettings,
    std::shared_ptr<cs::core::SolarSystem>               solarSystem)
    : mPluginSettings(std::move(pluginSettings))
    , mSolarSystem(std::move(solarSystem)*/) {
    // Add to scenegraph.
    /*VistaSceneGraph* pSG = GetVistaSystem()->GetGraphicsManager()->GetSceneGraph();
    mGLNode.reset(pSG->NewOpenGLNode(pSG->GetRoot(), this));
    VistaOpenSGMaterialTools::SetSortKeyOnSubtree(
    mGLNode.get(), static_cast<int>(cs::utils::DrawOrder::eTransparentItems) - 1);*/
}

////////////////////////////////////////////////////////////////////////////////////////////////////

Arrows::~Arrows() {
    /*VistaSceneGraph* pSG = GetVistaSystem()->GetGraphicsManager()->GetSceneGraph();
    pSG->GetRoot()->DisconnectChild(mGLNode.get());*/
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Arrows::update(double tTime) {
    return;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

/*void Arrows::setTargetName(std::string objectName) {
  mTargetName = std::move(objectName);
}*/

////////////////////////////////////////////////////////////////////////////////////////////////////

/*void Arrows::setParentName(std::string objectName) {
  mParentName = std::move(objectName);
}*/

////////////////////////////////////////////////////////////////////////////////////////////////////

/*std::string const& Arrows::getTargetName() const {
  return mTargetName;
}*/

////////////////////////////////////////////////////////////////////////////////////////////////////

/*std::string const& Arrows::getParentName() const {
  return mParentName;
}*/

////////////////////////////////////////////////////////////////////////////////////////////////////

bool Arrows::Do() {
    return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool Arrows::GetBoundingBox(VistaBoundingBox& /*bb*/) {
  return false;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Arrows::setEnabled(bool value) {
    mEnabled = std::move(value);
}

} // namespace csp::coordinatearows
