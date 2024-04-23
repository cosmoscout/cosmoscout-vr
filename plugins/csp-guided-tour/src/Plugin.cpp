////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "Plugin.hpp"
#include "logger.hpp"


#include "../../../src/cs-core/GraphicsEngine.hpp"
#include "../../../src/cs-core/GuiManager.hpp"
#include "../../../src/cs-core/InputManager.hpp"
#include "../../../src/cs-core/SolarSystem.hpp"
#include "../../../src/cs-core/TimeControl.hpp"
#include "../../../src/cs-utils/convert.hpp"

#include <VistaKernel/GraphicsManager/VistaOpenGLNode.h>
#include <VistaKernel/GraphicsManager/VistaSceneGraph.h>
#include <VistaKernel/GraphicsManager/VistaTransformNode.h>
#include <VistaKernel/VistaSystem.h>
#include <VistaKernelOpenSGExt/VistaOpenSGMaterialTools.h>
#include <glm/gtc/type_ptr.hpp>
////////////////////////////////////////////////////////////////////////////////////////////////////

EXPORT_FN cs::core::PluginBase* create() {
  return new csp::guidedtour::Plugin;
}
////////////////////////////////////////////////////////////////////////////////////////////////////

EXPORT_FN void destroy(cs::core::PluginBase* pluginBase) {
  delete pluginBase;
}
////////////////////////////////////////////////////////////////////////////////////////////////////

namespace csp::guidedtour {

void from_json(nlohmann::json const& j, Plugin::Settings::CheckPointSettings& o) {
  cs::core::Settings::deserialize(j, "object", o.mObject);
  cs::core::Settings::deserialize(j, "longitude", o.mLongitude);
  cs::core::Settings::deserialize(j, "latitude", o.mLatitude);
  cs::core::Settings::deserialize(j, "elevation", o.mElevation);
  cs::core::Settings::deserialize(j, "scale", o.mScale);
  cs::core::Settings::deserialize(j, "width", o.mWidth);
  cs::core::Settings::deserialize(j, "height", o.mHeight);
  cs::core::Settings::deserialize(j, "file", o.mFile);
}

void to_json(nlohmann::json& j, Plugin::Settings::CheckPointSettings const& o) {
  cs::core::Settings::serialize(j, "object", o.mObject);
  cs::core::Settings::serialize(j, "longitude", o.mLongitude);
  cs::core::Settings::serialize(j, "latitude", o.mLatitude);
  cs::core::Settings::serialize(j, "elevation", o.mElevation);
  cs::core::Settings::serialize(j, "scale", o.mScale);
  cs::core::Settings::serialize(j, "width", o.mWidth);
  cs::core::Settings::serialize(j, "height", o.mHeight);
  cs::core::Settings::serialize(j, "file", o.mFile);
}

// From_Json to_json hinzugeüfgt.
void from_json(nlohmann::json const& j, Plugin::Settings::TourSettings& o) {
  cs::core::Settings::deserialize(j, "name", o.mName);
  cs::core::Settings::deserialize(j, "planet", o.mPlanet);
  cs::core::Settings::deserialize(j, "startPosition", o.mTourPositionStart);
  cs::core::Settings::deserialize(j, "startRotation", o.mTourRotationStart);
  cs::core::Settings::deserialize(j, "checkpoints", o.mCheckpoints);
}

void to_json(nlohmann::json& j, Plugin::Settings::TourSettings const& o) {
  cs::core::Settings::serialize(j, "name", o.mName);
  cs::core::Settings::serialize(j, "planet", o.mPlanet);
  cs::core::Settings::serialize(j, "startPosition", o.mTourPositionStart);
  cs::core::Settings::serialize(j, "startRotation", o.mTourRotationStart);
  cs::core::Settings::serialize(j, "checkpoints", o.mCheckpoints);
}

/////////////////////////////////////////////////////////////////////From_Json to_json
/// hinzugeüfgt.//
// cs::core::Settings::serialize(j, "checkpoints", o.mCPItems);

void from_json(nlohmann::json const& j, Plugin::Settings& o) {
  cs::core::Settings::deserialize(j, "tours", o.mTours);
}

void to_json(nlohmann::json& j, Plugin::Settings const& o) {
  cs::core::Settings::serialize(j, "tours", o.mTours);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::init() {
  logger().info("Loading plugin...");

  mOnLoadConnection = mAllSettings->onLoad().connect([this]() { onLoad(); });
  mOnSaveConnection = mAllSettings->onSave().connect([this]() { onSave(); });

  mGuiManager->executeJavascriptFile("../share/resources/gui/js/csp-guided-tour.js");

  mGuiManager->addPluginTabToSideBarFromHTML(
      "Guided Tour", "flag", "../share/resources/gui/csp-guided-tour-tab.html");

  mGuiManager->getGui()->registerCallback("guidedTours.reset",
      "Call this to reset all Checkpoints of the current tour.", std::function([this] {
        for (auto& tour : mPluginSettings.mTours) {
          unload(mPluginSettings);
          for (auto& checkpoint : tour.mCheckpoints) {

            checkpoint.mIsVisited = false;
          }
        }

        unload(mPluginSettings);
        mGuiManager->getGui()->callJavascript("CosmoScout.guidedTours.resetAll");
      }));
  mGuiManager->getGui()->registerCallback("guidedTours.loadTour",
      "Call this to load the specified tour.",
      std::function([this](std::string&& tourName) { mCurrentTour = tourName; }));

  // Load initial settings.
  onLoad();

  logger().info("Loading done.");
}
////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::update() {
  if (mCurrentTour != "") {
    unload(mPluginSettings); // Delete Checkpoints statt unload
    if (mCurrentTour != "none") {
      loadCheckpoints();
    }
    mCurrentTour = "";
  }
  // Rotate the space items to face the observer.
  for (auto& item : mCPItems) {
    auto object = mSolarSystem->getObject(item.mObjectName);

    if (object) {
      auto scale = mSolarSystem->getScaleBasedOnObserverDistance(
          object, item.mPosition, item.mScale, mAllSettings->mGraphics.pWorldUIScale.get());
      auto rotation = mSolarSystem->getRotationToObserver(object, item.mPosition, true);

      auto transform = object->getObserverRelativeTransform(item.mPosition, rotation, scale);
      item.mAnchor->SetTransform(glm::value_ptr(transform), true);
    }
  }
}

void Plugin::deInit() {
  onSave();
  unload(mPluginSettings);
  mAllSettings->onLoad().disconnect(mOnLoadConnection);
  mAllSettings->onSave().disconnect(mOnSaveConnection);

  mGuiManager->removePluginTab("Guided Tour");
}

void Plugin::onLoad() {
  logger().info("onLoad Start");
  auto oldSettings = mPluginSettings;
  from_json(mAllSettings->mPlugins.at("csp-guided-tour"), mPluginSettings);
  for (auto const& tour : mPluginSettings.mTours) {
    mGuiManager->getGui()->callJavascript("CosmoScout.guidedTours.add", tour.mName);
  }
  //
  loadCheckpoints();
  logger().info("onLoad End");
}

void Plugin::onSave() {
  mAllSettings->mPlugins["csp-guided-tour"] = mPluginSettings;
}

void Plugin::unload(Settings const& pluginSettings) {
  auto* pSG = GetVistaSystem()->GetGraphicsManager()->GetSceneGraph();
  for (auto const& item : mCPItems) {
    pSG->GetRoot()->DisconnectChild(item.mAnchor.get());
    mInputManager->unregisterSelectable(item.mGuiNode.get());
  }
  mCPItems.clear();
}
void Plugin::loadCheckpoints() {

  for (auto const& tour : mPluginSettings.mTours) {
    if (tour.mName == mCurrentTour) {

      logger().info(tour.mPlanet);
      mGuiManager->getGui()->callJavascript(
          "CosmoScout.callbacks.navigation.setBody", tour.mPlanet);

      // Übergeben der Position an JavaScript
      auto posIterator = tour.mTourPositionStart.begin();
      auto firstPos    = *posIterator;
      ++posIterator;
      auto secondPos = *posIterator;
      ++posIterator;
      auto thirdPos = *posIterator;
           auto rotIterator = tour.mTourRotationStart.begin();
      auto firstRot    = *rotIterator;
      ++rotIterator;
      auto secondRot = *rotIterator;
      ++rotIterator;
      auto thirdRot = *rotIterator;
      ++rotIterator;
      auto fourthRot = *rotIterator;

      mGuiManager->getGui()->callJavascript(
          "CosmoScout.callbacks.navigation.setPosition", firstPos, secondPos, thirdPos);

      
      //mGuiManager->getGui()->callJavascript(
      //    "CosmoScout.callbacks.navigation.setRotation", firstRot, secondRot, thirdRot, fourthRot);

      for (auto const& settings : tour.mCheckpoints) {

        auto object = mSolarSystem->getObject(settings.mObject);

        CPItem item;
        item.mObjectName = settings.mObject;

        auto* pSG = GetVistaSystem()->GetGraphicsManager()->GetSceneGraph();
        item.mAnchor.reset(pSG->NewTransformNode(pSG->GetRoot()));
        item.mScale = settings.mScale;

        glm::dvec2 lngLat(settings.mLongitude, settings.mLatitude);
        lngLat         = cs::utils::convert::toRadians(lngLat);
        auto radii     = object->getRadii();
        item.mPosition = cs::utils::convert::toCartesian(lngLat, radii, settings.mElevation);

        item.mGuiArea =
            std::make_unique<cs::gui::WorldSpaceGuiArea>(settings.mWidth, settings.mHeight);
        item.mTransform.reset(pSG->NewTransformNode(item.mAnchor.get()));
        item.mTransform->Scale(0.001F * static_cast<float>(item.mGuiArea->getWidth()),
            0.001F * static_cast<float>(item.mGuiArea->getHeight()), 1.F);
        item.mTransform->Rotate(
            VistaAxisAndAngle(VistaVector3D(0.0, 1.0, 0.0), -glm::pi<float>() / 2.F));
        item.mGuiNode.reset(pSG->NewOpenGLNode(item.mTransform.get(), item.mGuiArea.get()));
        mInputManager->registerSelectable(item.mGuiNode.get());
        VistaOpenSGMaterialTools::SetSortKeyOnSubtree(
            item.mGuiNode.get(), static_cast<int>(cs::utils::DrawOrder::eTransparentItems));
        item.mGuiItem = std::make_unique<cs::gui::GuiItem>("file://" + settings.mFile);

        item.mGuiItem->registerCallback(
            "setVisitedCpp", "Sets the isVisited Boolean.", std::function<void()>([&]() {
              settings.mIsVisited = true;

              int cpCount   = tour.mCheckpoints.size();
              int cpVisited = 0;
              for (auto const& settings : tour.mCheckpoints) {
                if (settings.mIsVisited) {
                  cpVisited++;
                }
              }
              mGuiManager->getGui()->callJavascript(
                  "CosmoScout.guidedTours.setProgress", tour.mName, cpCount, cpVisited);
            }));
        item.mGuiArea->addItem(item.mGuiItem.get());
        item.mGuiItem->setCursorChangeCallback(
            [](cs::gui::Cursor c) { cs::core::GuiManager::setCursor(c); });
        item.mGuiItem->waitForFinishedLoading();
        item.mGuiItem->callJavascript("setVisited", settings.mIsVisited);
        mCPItems.emplace_back(std::move(item));
      }
    }
  }
}

} // namespace csp::guidedtour