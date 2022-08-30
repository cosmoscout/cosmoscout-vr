////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

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
  return new csp::customwebui::Plugin;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

EXPORT_FN void destroy(cs::core::PluginBase* pluginBase) {
  delete pluginBase;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace csp::customwebui {

////////////////////////////////////////////////////////////////////////////////////////////////////

void from_json(nlohmann::json const& j, Plugin::Settings::GuiItem& o) {
  cs::core::Settings::deserialize(j, "name", o.mName);
  cs::core::Settings::deserialize(j, "icon", o.mIcon);
  cs::core::Settings::deserialize(j, "html", o.mHTML);
}

void to_json(nlohmann::json& j, Plugin::Settings::GuiItem const& o) {
  cs::core::Settings::serialize(j, "name", o.mName);
  cs::core::Settings::serialize(j, "icon", o.mIcon);
  cs::core::Settings::serialize(j, "html", o.mHTML);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void from_json(nlohmann::json const& j, Plugin::Settings::SpaceItem& o) {
  cs::core::Settings::deserialize(j, "object", o.mObject);
  cs::core::Settings::deserialize(j, "longitude", o.mLongitude);
  cs::core::Settings::deserialize(j, "latitude", o.mLatitude);
  cs::core::Settings::deserialize(j, "elevation", o.mElevation);
  cs::core::Settings::deserialize(j, "scale", o.mScale);
  cs::core::Settings::deserialize(j, "width", o.mWidth);
  cs::core::Settings::deserialize(j, "height", o.mHeight);
  cs::core::Settings::deserialize(j, "html", o.mHTML);
}

void to_json(nlohmann::json& j, Plugin::Settings::SpaceItem const& o) {
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
  cs::core::Settings::deserialize(j, "sidebar-items", o.mSideBarItems);
  cs::core::Settings::deserialize(j, "window-items", o.mWindowItems);
  cs::core::Settings::deserialize(j, "space-items", o.mSpaceItems);
}

void to_json(nlohmann::json& j, Plugin::Settings const& o) {
  cs::core::Settings::serialize(j, "sidebar-items", o.mSideBarItems);
  cs::core::Settings::serialize(j, "window-items", o.mWindowItems);
  cs::core::Settings::serialize(j, "space-items", o.mSpaceItems);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool Plugin::Settings::GuiItem::operator==(Plugin::Settings::GuiItem const& other) const {
  return mName == other.mName && mIcon == other.mIcon && mHTML == other.mHTML;
}

bool Plugin::Settings::SpaceItem::operator==(Plugin::Settings::SpaceItem const& other) const {
  return mObject == other.mObject && mLongitude == other.mLongitude &&
         mLatitude == other.mLatitude && mElevation == other.mElevation && mScale == other.mScale &&
         mWidth == other.mWidth && mHeight == other.mHeight && mHTML == other.mHTML;
}

bool Plugin::Settings::operator==(Plugin::Settings const& other) const {
  return mSideBarItems == other.mSideBarItems && mWindowItems == other.mWindowItems &&
         mSpaceItems == other.mSpaceItems;
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
  for (auto& item : mSpaceItems) {
    auto object = mSolarSystem->getObject(item.mObjectName);

    if (object) {
      auto scale = mSolarSystem->getScaleBasedOnObserverDistance(
          object, item.mPosition, item.mScale, mAllSettings->mGraphics.pWorldUIScale.get());
      auto rotation = mSolarSystem->getRotationToObserver(object, item.mPosition, false);

      auto transform = object->getObserverRelativeTransform(item.mPosition, rotation, scale);
      item.mAnchor->SetTransform(glm::value_ptr(transform), true);
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::deInit() {

  // Save settings as this plugin may get reloaded.
  onSave();

  // Remove all items.
  unload(mPluginSettings);

  mAllSettings->onLoad().disconnect(mOnLoadConnection);
  mAllSettings->onSave().disconnect(mOnSaveConnection);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::onLoad() {

  // Store current settings to check whether anything changed due to loading.
  auto oldSettings = mPluginSettings;

  // Read settings from JSON.
  from_json(mAllSettings->mPlugins.at("csp-custom-web-ui"), mPluginSettings);

  // We simply reload everything if anything changed.
  if (mPluginSettings != oldSettings) {

    // First remove everything we created before.
    unload(oldSettings);

    // Then add all new sidebar tabs.
    for (auto const& settings : mPluginSettings.mSideBarItems) {
      mGuiManager->addPluginTabToSideBar(settings.mName, settings.mIcon, settings.mHTML);
    }

    // Then add all window items.
    for (size_t i(0); i < mPluginSettings.mWindowItems.size(); ++i) {
      auto const& settings = mPluginSettings.mWindowItems[i];

      std::string callback = "customWebUI.toggleWindow" + std::to_string(i);
      std::string id       = "customWebUIWindow" + std::to_string(i);

      // Register a callback to toggle the window.
      mGuiManager->getGui()->registerCallback(callback,
          "Toggles the custom window '" + settings.mName + "'.", std::function([this, id]() {
            mGuiManager->getGui()->executeJavascript(
                "document.querySelector('#" + id + "').classList.toggle('visible')");
          }));

      // Add a timeline button to toggle the window.
      mGuiManager->addTimelineButton(settings.mName, settings.mIcon, callback);

      // Add the window itself.
      std::string windowMarkup = R"(
        <div id="%ID%" class="draggable-window resizable-window auto-hide-header">
          <div class="window-wrapper">
            <div class="window-header">
              <span class="window-title"><i class="material-icons">%ICON%</i><span>%NAME%</span></span>
              <a class="btn light-glass" data-action="close" data-toggle="tooltip" title="Close">
                <i class="material-icons">close</i>
              </a>
            </div>
            <div class="window-content">
              %CONTENT%
            </div>
          </div>
        </div>
      )";

      cs::utils::replaceString(windowMarkup, "%ID%", id);
      cs::utils::replaceString(windowMarkup, "%ICON%", settings.mIcon);
      cs::utils::replaceString(windowMarkup, "%NAME%", settings.mName);
      cs::utils::replaceString(windowMarkup, "%CONTENT%", settings.mHTML);

      mGuiManager->getGui()->callJavascript("CosmoScout.gui.addHtml", windowMarkup);
      mGuiManager->getGui()->callJavascript("CosmoScout.gui.initDraggableWindows");
    }

    // Then add all space items.
    auto* pSG = GetVistaSystem()->GetGraphicsManager()->GetSceneGraph();
    for (auto const& settings : mPluginSettings.mSpaceItems) {

      auto object = mSolarSystem->getObject(settings.mObject);

      if (!object) {
        continue;
      }

      SpaceItem item;
      item.mObjectName = settings.mObject;

      // Create a VistaTransformNode to attach our gui element to.
      item.mAnchor.reset(pSG->NewTransformNode(pSG->GetRoot()));
      item.mScale = settings.mScale;

      // Compute the cartesian position of the gui item.
      glm::dvec2 lngLat(settings.mLongitude, settings.mLatitude);
      lngLat         = cs::utils::convert::toRadians(lngLat);
      auto radii     = object->getRadii();
      item.mPosition = cs::utils::convert::toCartesian(lngLat, radii, settings.mElevation);

      // Create the WorldSpaceGuiArea for the gui element.
      item.mGuiArea =
          std::make_unique<cs::gui::WorldSpaceGuiArea>(settings.mWidth, settings.mHeight);

      // Create a TransformNode to attach the gui element to.
      item.mTransform.reset(pSG->NewTransformNode(item.mAnchor.get()));
      item.mTransform->Scale(0.001F * static_cast<float>(item.mGuiArea->getWidth()),
          0.001F * static_cast<float>(item.mGuiArea->getHeight()), 1.F);
      item.mTransform->Rotate(
          VistaAxisAndAngle(VistaVector3D(0.0, 1.0, 0.0), -glm::pi<float>() / 2.F));

      // Attach an OpenGLNode to the TransformNode containing our WorldSpaceGuiArea.
      item.mGuiNode.reset(pSG->NewOpenGLNode(item.mTransform.get(), item.mGuiArea.get()));
      mInputManager->registerSelectable(item.mGuiNode.get());
      VistaOpenSGMaterialTools::SetSortKeyOnSubtree(
          item.mGuiNode.get(), static_cast<int>(cs::utils::DrawOrder::eTransparentItems));

      // Add the GuiItem to our WorldSpaceGuiArea.
      item.mGuiItem = std::make_unique<cs::gui::GuiItem>(
          "file://../share/resources/gui/custom-web-ui-simple.html");
      item.mGuiArea->addItem(item.mGuiItem.get());
      item.mGuiItem->setCursorChangeCallback(
          [](cs::gui::Cursor c) { cs::core::GuiManager::setCursor(c); });
      item.mGuiItem->waitForFinishedLoading();
      item.mGuiItem->callJavascript("setContent", settings.mHTML);

      // Store it.
      mSpaceItems.emplace_back(std::move(item));
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::onSave() {
  mAllSettings->mPlugins["csp-custom-web-ui"] = mPluginSettings;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::unload(Settings const& pluginSettings) {
  // Remove all sidebar tabs.
  for (auto const& settings : pluginSettings.mSideBarItems) {
    mGuiManager->removePluginTab(settings.mName);
  }

  // Remove all window items.
  for (size_t i(0); i < pluginSettings.mWindowItems.size(); ++i) {
    mGuiManager->getGui()->unregisterCallback("customWebUI.toggleWindow" + std::to_string(i));
    mGuiManager->removeTimelineButton(pluginSettings.mWindowItems[i].mName);

    std::string id = "customWebUIWindow" + std::to_string(i);
    mGuiManager->getGui()->executeJavascript("document.querySelector('#" + id + "').remove()");
  }

  // Remove all space items.
  auto* pSG = GetVistaSystem()->GetGraphicsManager()->GetSceneGraph();
  for (auto const& item : mSpaceItems) {
    pSG->GetRoot()->DisconnectChild(item.mAnchor.get());
    mInputManager->unregisterSelectable(item.mGuiNode.get());
  }
  mSpaceItems.clear();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::customwebui
