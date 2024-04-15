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

void from_json(nlohmann::json const& j, Plugin::Settings::CPItem& o) {
  cs::core::Settings::deserialize(j, "object", o.mObject);
  cs::core::Settings::deserialize(j, "longitude", o.mLongitude);
  cs::core::Settings::deserialize(j, "latitude", o.mLatitude);
  cs::core::Settings::deserialize(j, "elevation", o.mElevation);
  cs::core::Settings::deserialize(j, "scale", o.mScale);
  cs::core::Settings::deserialize(j, "width", o.mWidth);
  cs::core::Settings::deserialize(j, "height", o.mHeight);
  cs::core::Settings::deserialize(j, "html", o.mHTML);
}

void to_json(nlohmann::json& j, Plugin::Settings::CPItem const& o) {
  cs::core::Settings::serialize(j, "object", o.mObject);
  cs::core::Settings::serialize(j, "longitude", o.mLongitude);
  cs::core::Settings::serialize(j, "latitude", o.mLatitude);
  cs::core::Settings::serialize(j, "elevation", o.mElevation);
  cs::core::Settings::serialize(j, "scale", o.mScale);
  cs::core::Settings::serialize(j, "width", o.mWidth);
  cs::core::Settings::serialize(j, "height", o.mHeight);
  cs::core::Settings::serialize(j, "html", o.mHTML);
}
////////////////////////////////////////////////////////////////////////////////////////////////////

void from_json(nlohmann::json const& j, Plugin::Settings& o) {
  cs::core::Settings::deserialize(j, "cp-items", o.mCPItems);
}

void to_json(nlohmann::json& j, Plugin::Settings const& o) {
  cs::core::Settings::serialize(j, "cp-items", o.mCPItems);
}
////////////////////////////////////////////////////////////////////////////////////////////////////

bool Plugin::Settings::CPItem::operator==(Plugin::Settings::CPItem const& other) const {
  return mObject == other.mObject && mLongitude == other.mLongitude &&
         mLatitude == other.mLatitude && mElevation == other.mElevation && mScale == other.mScale &&
         mWidth == other.mWidth && mHeight == other.mHeight && mHTML == other.mHTML;
}

bool Plugin::Settings::operator==(Plugin::Settings const& other) const {
  return mCPItems == other.mCPItems;
}

bool Plugin::Settings::operator!=(Plugin::Settings const& other) const {
  return !(*this == other);
}
////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::init() {
  logger().info("Loading plugin...");

  mOnLoadConnection = mAllSettings->onLoad().connect([this]() { onLoad(); });
  mOnSaveConnection = mAllSettings->onSave().connect([this]() { onSave(); });
  // Load initial settings.

  onLoad();

  logger().info("Loading done.");
}
////////////////////////////////////////////////////////////////////////////////////////////////////


void Plugin::update() {
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
}

void Plugin::onLoad() {
  auto oldSettings = mPluginSettings;
  from_json(mAllSettings->mPlugins.at("csp-guided-tour"), mPluginSettings);

  if (mPluginSettings != oldSettings) {
    unload(oldSettings);

    for (auto const& settings : mPluginSettings.mCPItems) {
      auto object = mSolarSystem->getObject(settings.mObject);

      if (!object) {
        continue;
      }

      CPItem item;
      item.mObjectName = settings.mObject;

      auto* pSG = GetVistaSystem()->GetGraphicsManager()->GetSceneGraph();

      item.mAnchor.reset(pSG->NewTransformNode(pSG->GetRoot()));
      item.mScale = settings.mScale;

      glm::dvec2 lngLat(settings.mLongitude, settings.mLatitude);
      lngLat = cs::utils::convert::toRadians(lngLat);
      auto radii = object->getRadii();
      item.mPosition = cs::utils::convert::toCartesian(lngLat, radii, settings.mElevation);

      item.mGuiArea = std::make_unique<cs::gui::WorldSpaceGuiArea>(settings.mWidth, settings.mHeight);

      item.mTransform.reset(pSG->NewTransformNode(item.mAnchor.get()));
      item.mTransform->Scale(0.001F * static_cast<float>(item.mGuiArea->getWidth()),
          0.001F * static_cast<float>(item.mGuiArea->getHeight()), 1.F);
      item.mTransform->Rotate(
          VistaAxisAndAngle(VistaVector3D(0.0, 1.0, 0.0), -glm::pi<float>() / 2.F));

      item.mGuiNode.reset(pSG->NewOpenGLNode(item.mTransform.get(), item.mGuiArea.get()));
      mInputManager->registerSelectable(item.mGuiNode.get());
      VistaOpenSGMaterialTools::SetSortKeyOnSubtree(
          item.mGuiNode.get(), static_cast<int>(cs::utils::DrawOrder::eTransparentItems));

      item.mGuiItem = std::make_unique<cs::gui::GuiItem>(
          "file://../share/resources/gui/guided-tour-simple.html");
      item.mGuiArea->addItem(item.mGuiItem.get());
      item.mGuiItem->setCursorChangeCallback(
          [](cs::gui::Cursor c) { cs::core::GuiManager::setCursor(c); });
      item.mGuiItem->waitForFinishedLoading();
      item.mGuiItem->callJavascript("setContent", settings.mHTML);

      mCPItems.emplace_back(std::move(item));
    }
  }
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

} // namespace csp::guidedtour
