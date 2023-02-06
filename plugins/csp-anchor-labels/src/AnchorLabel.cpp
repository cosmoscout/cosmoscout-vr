////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "AnchorLabel.hpp"

#include "logger.hpp"

#include "../../../src/cs-core/GuiManager.hpp"
#include "../../../src/cs-core/InputManager.hpp"
#include "../../../src/cs-core/SolarSystem.hpp"
#include "../../../src/cs-core/TimeControl.hpp"
#include "../../../src/cs-gui/GuiItem.hpp"
#include "../../../src/cs-gui/WorldSpaceGuiArea.hpp"
#include "../../../src/cs-scene/CelestialObject.hpp"

#include <GL/glew.h>
#include <VistaKernel/GraphicsManager/VistaGraphicsManager.h>
#include <VistaKernel/GraphicsManager/VistaOpenGLNode.h>
#include <VistaKernel/GraphicsManager/VistaSceneGraph.h>
#include <VistaKernel/GraphicsManager/VistaTransformNode.h>
#include <VistaKernel/VistaSystem.h>
#include <VistaKernelOpenSGExt/VistaOpenSGMaterialTools.h>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/component_wise.hpp>
#include <glm/gtx/norm.hpp>
#include <utility>

namespace csp::anchorlabels {

////////////////////////////////////////////////////////////////////////////////////////////////////

AnchorLabel::AnchorLabel(std::string const&           name,
    std::shared_ptr<const cs::scene::CelestialObject> object,
    std::shared_ptr<Plugin::Settings>                 pluginSettings,
    std::shared_ptr<cs::core::SolarSystem>            solarSystem,
    std::shared_ptr<cs::core::GuiManager>             guiManager,
    std::shared_ptr<cs::core::InputManager>           inputManager)
    : mObject(std::move(object))
    , mPluginSettings(std::move(pluginSettings))
    , mSolarSystem(std::move(solarSystem))
    , mGuiManager(std::move(guiManager))
    , mInputManager(std::move(inputManager))
    , mGuiArea(std::make_unique<cs::gui::WorldSpaceGuiArea>(150, 30))
    , mGuiItem(
          std::make_unique<cs::gui::GuiItem>("file://../share/resources/gui/anchor_label.html")) {
  auto* sceneGraph = GetVistaSystem()->GetGraphicsManager()->GetSceneGraph();

  mObjectTransform.reset(sceneGraph->NewTransformNode(sceneGraph->GetRoot()));

  mGuiTransform.reset(sceneGraph->NewTransformNode(mObjectTransform.get()));
  mGuiTransform->SetScale(1.2F,
      1.2F * static_cast<float>(mGuiArea->getHeight()) / static_cast<float>(mGuiArea->getWidth()),
      1.0F);
  mGuiTransform->SetTranslation(
      0.0F, static_cast<float>(mPluginSettings->mLabelOffset.get()), 0.0F);
  mGuiTransform->Rotate(VistaAxisAndAngle(VistaVector3D(0.0, 1.0, 0.0), -glm::pi<float>() / 2.F));

  mGuiNode.reset(sceneGraph->NewOpenGLNode(mGuiTransform.get(), mGuiArea.get()));
  mInputManager->registerSelectable(mGuiNode.get());

  mGuiArea->addItem(mGuiItem.get());
  mGuiArea->setIgnoreDepth(false);

  mGuiItem->setCanScroll(false);
  mGuiItem->waitForFinishedLoading();

  mGuiItem->registerCallback(
      "flyToBody", "Makes the observer fly to the planet marked by this anchor label.", [this] {
        mSolarSystem->flyObserverTo(mObject->getCenterName(), mObject->getFrameName(), 5.0);
        mGuiManager->showNotification("Travelling", "to " + mObject->getCenterName(), "send");
      });

  mGuiItem->callJavascript("setLabelText", name);

  mOffsetConnection = mPluginSettings->mLabelOffset.connect([this](double newOffset) {
    mGuiTransform->SetTranslation(0.0F, static_cast<float>(newOffset), 0.0F);
  });
}

////////////////////////////////////////////////////////////////////////////////////////////////////

AnchorLabel::~AnchorLabel() {
  mGuiItem->unregisterCallback("flyToBody");

  mGuiTransform->DisconnectChild(mGuiNode.get());
  auto* sceneGraph = GetVistaSystem()->GetGraphicsManager()->GetSceneGraph();
  sceneGraph->GetRoot()->DisconnectChild(mObjectTransform.get());

  mInputManager->unregisterSelectable(mGuiNode.get());

  mPluginSettings->mLabelOffset.disconnect(mOffsetConnection);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void AnchorLabel::update() {
  mRelativeAnchorPosition = mObject->getObserverRelativePosition();

  if (mObject->getIsOrbitVisible()) {
    double distanceToObserver = distanceToCamera();

    double const scaleFactor = 0.05;
    double       scale       = glm::pow(distanceToObserver, mPluginSettings->mDepthScale.get()) *
                   mPluginSettings->mLabelScale.get() * scaleFactor;

    auto       mat    = mObject->getObserverRelativeTransform();
    glm::dvec3 pos    = mat[3];
    glm::dvec3 y      = glm::dvec4(0, 1, 0, 0);
    glm::dvec3 camDir = -glm::normalize(pos);

    glm::dvec3 z = glm::cross(y, camDir);
    glm::dvec3 x = glm::cross(y, z);

    x = glm::normalize(x);
    y = glm::normalize(y);
    z = glm::normalize(z);

    auto       rotation = glm::toQuat(glm::dmat3(x, y, z));
    double     angle    = glm::angle(rotation);
    glm::dvec3 axis     = glm::axis(rotation);

    mat = glm::dmat4(1.0);
    mat = glm::translate(mat, pos);
    mat = glm::rotate(mat, angle, axis);
    mat = glm::scale(mat, glm::dvec3(scale, scale, scale));

    mObjectTransform->SetTransform(glm::value_ptr(mat), true);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

glm::dvec4 AnchorLabel::getScreenSpaceBB() const {
  double const width =
      mPluginSettings->mLabelScale.get() * static_cast<double>(mGuiArea->getWidth()) * 0.0005;
  double const height =
      mPluginSettings->mLabelScale.get() * static_cast<double>(mGuiArea->getHeight()) * 0.0005;

  auto const screenPos = (mRelativeAnchorPosition.xyz() / mRelativeAnchorPosition.z).xy();

  double const x = screenPos.x - (width / 2.0);
  double const y = screenPos.y - (height / 2.0);

  return {x, y, width, height};
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void AnchorLabel::enable() const {
  mGuiItem->setIsEnabled(true);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void AnchorLabel::disable() const {
  mGuiItem->setIsEnabled(false);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::shared_ptr<const cs::scene::CelestialObject> const& AnchorLabel::getObject() const {
  return mObject;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool AnchorLabel::shouldBeHidden() const {
  return !mObject->getIsOrbitVisible();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

double AnchorLabel::bodySize() const {
  return glm::compMax(mObject->getRadii());
}

////////////////////////////////////////////////////////////////////////////////////////////////////

double AnchorLabel::distanceToCamera() const {
  return glm::length(mRelativeAnchorPosition);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void AnchorLabel::setSortKey(int key) const {
  VistaOpenSGMaterialTools::SetSortKeyOnSubtree(mGuiTransform.get(), key);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::anchorlabels
