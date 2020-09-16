////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "AnchorLabel.hpp"

#include "logger.hpp"

#include "../../../src/cs-core/GuiManager.hpp"
#include "../../../src/cs-core/InputManager.hpp"
#include "../../../src/cs-core/SolarSystem.hpp"
#include "../../../src/cs-core/TimeControl.hpp"
#include "../../../src/cs-gui/GuiItem.hpp"
#include "../../../src/cs-gui/WorldSpaceGuiArea.hpp"
#include "../../../src/cs-scene/CelestialAnchorNode.hpp"
#include "../../../src/cs-scene/CelestialBody.hpp"
#include "../../../src/cs-utils/FrameTimings.hpp"

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

AnchorLabel::AnchorLabel(cs::scene::CelestialBody const* const body,
    std::shared_ptr<Plugin::Settings>                          pluginSettings,
    std::shared_ptr<cs::core::SolarSystem>                     solarSystem,
    std::shared_ptr<cs::core::GuiManager>                      guiManager,
    std::shared_ptr<cs::core::TimeControl>                     timeControl,
    std::shared_ptr<cs::core::InputManager>                    inputManager)
    : mBody(body)
    , mPluginSettings(std::move(pluginSettings))
    , mSolarSystem(std::move(solarSystem))
    , mGuiManager(std::move(guiManager))
    , mTimeControl(std::move(timeControl))
    , mInputManager(std::move(inputManager))
    , mGuiArea(std::make_unique<cs::gui::WorldSpaceGuiArea>(120, 30)) // NOLINT
    , mGuiItem(
          std::make_unique<cs::gui::GuiItem>("file://../share/resources/gui/anchor_label.html")) {
  auto* sceneGraph = GetVistaSystem()->GetGraphicsManager()->GetSceneGraph();

  mAnchor = std::make_shared<cs::scene::CelestialAnchorNode>(sceneGraph->GetRoot(),
      sceneGraph->GetNodeBridge(), "", mBody->getCenterName(), mBody->getFrameName());
  mAnchor->setAnchorPosition(mBody->getAnchorPosition());

  mGuiTransform.reset(sceneGraph->NewTransformNode(mAnchor.get()));
  mGuiTransform->SetScale(1.0F,
      static_cast<float>(mGuiArea->getHeight()) / static_cast<float>(mGuiArea->getWidth()), 1.0F);
  mGuiTransform->SetTranslation(
      0.0F, static_cast<float>(mPluginSettings->mLabelOffset.get()), 0.0F);
  mGuiTransform->Rotate(VistaAxisAndAngle(VistaVector3D(0.0, 1.0, 0.0), -glm::pi<float>() / 2.F));

  mGuiNode.reset(sceneGraph->NewOpenGLNode(mGuiTransform.get(), mGuiArea.get()));
  mInputManager->registerSelectable(mGuiNode.get());

  mGuiArea->addItem(mGuiItem.get());
  mGuiArea->setUseLinearDepthBuffer(true);
  mGuiArea->setIgnoreDepth(false);

  mGuiItem->setCanScroll(false);
  mGuiItem->waitForFinishedLoading();

  mGuiItem->registerCallback(
      "flyToBody", "Makes the observer fly to the planet marked by this anchor label.", [this] {
        mSolarSystem->flyObserverTo(mBody->getCenterName(), mBody->getFrameName(), 5.0);
        mGuiManager->showNotification("Travelling", "to " + mBody->getCenterName(), "send");
      });

  mGuiItem->callJavascript("setLabelText", mBody->getCenterName());

  mOffsetConnection = mPluginSettings->mLabelOffset.connect([this](double newOffset) {
    mGuiTransform->SetTranslation(0.0F, static_cast<float>(newOffset), 0.0F);
  });
}

////////////////////////////////////////////////////////////////////////////////////////////////////

AnchorLabel::~AnchorLabel() {
  mGuiItem->unregisterCallback("flyToBody");

  mGuiTransform->DisconnectChild(mGuiNode.get());
  auto* sceneGraph = GetVistaSystem()->GetGraphicsManager()->GetSceneGraph();
  sceneGraph->GetRoot()->DisconnectChild(mGuiTransform.get());

  mInputManager->unregisterSelectable(mGuiNode.get());

  mPluginSettings->mLabelOffset.disconnect(mOffsetConnection);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void AnchorLabel::update() {
  if (mBody->getIsInExistence()) {
    double simulationTime(mTimeControl->pSimulationTime.get());

    cs::scene::CelestialAnchor rawAnchor(mAnchor->getCenterName(), mAnchor->getFrameName());
    rawAnchor.setAnchorPosition(mAnchor->getAnchorPosition());

    try {
      mRelativeAnchorPosition =
          mSolarSystem->getObserver().getRelativePosition(simulationTime, rawAnchor);

      double distanceToObserver = distanceToCamera();

      double const scaleFactor = 0.05;
      double       scale       = mSolarSystem->getObserver().getAnchorScale();
      scale *= glm::pow(distanceToObserver, mPluginSettings->mDepthScale.get()) *
               mPluginSettings->mLabelScale.get() * scaleFactor;
      mAnchor->setAnchorScale(scale);

      auto observerTransform =
          rawAnchor.getRelativeTransform(simulationTime, mSolarSystem->getObserver());
      glm::dvec3 observerPos = observerTransform[3];
      glm::dvec3 y           = observerTransform * glm::dvec4(0, 1, 0, 0);
      glm::dvec3 camDir      = glm::normalize(observerPos);

      glm::dvec3 z = glm::cross(y, camDir);
      glm::dvec3 x = glm::cross(y, z);

      x = glm::normalize(x);
      y = glm::normalize(y);
      z = glm::normalize(z);

      auto rot = glm::toQuat(glm::dmat3(x, y, z));
      mAnchor->setAnchorRotation(rot);

      mAnchor->update(simulationTime, mSolarSystem->getObserver());

    } catch (std::exception const& e) {
      // Getting the relative transformation may fail due to insufficient SPICE data.
      logger().warn("AnchorLabel::update failed for '{}': {}", mBody->getCenterName(), e.what());
    }
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

std::string const& AnchorLabel::getCenterName() const {
  return mBody->getCenterName();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool AnchorLabel::shouldBeHidden() const {
  return !mBody->getIsInExistence();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

double AnchorLabel::bodySize() const {
  return glm::compMax(mBody->getRadii());
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
