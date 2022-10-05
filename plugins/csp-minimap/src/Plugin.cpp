////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "Plugin.hpp"
#include "logger.hpp"

#include "../../../src/cs-core/GuiManager.hpp"
#include "../../../src/cs-core/SolarSystem.hpp"
#include "../../../src/cs-utils/convert.hpp"

////////////////////////////////////////////////////////////////////////////////////////////////////

EXPORT_FN cs::core::PluginBase* create() {
  return new csp::minimap::Plugin;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

EXPORT_FN void destroy(cs::core::PluginBase* pluginBase) {
  delete pluginBase; // NOLINT(cppcoreguidelines-owning-memory)
}

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace csp::minimap {

////////////////////////////////////////////////////////////////////////////////////////////////////

// clang-format off

// NOLINTNEXTLINE
NLOHMANN_JSON_SERIALIZE_ENUM(Plugin::Settings::ProjectionType, {
    {Plugin::Settings::ProjectionType::eNone, nullptr},
    {Plugin::Settings::ProjectionType::eMercator, "mercator"},
    {Plugin::Settings::ProjectionType::eEquirectangular, "equirectangular"},
});

// NOLINTNEXTLINE
NLOHMANN_JSON_SERIALIZE_ENUM(Plugin::Settings::LayerType, {
   {Plugin::Settings::LayerType::eNone, nullptr},
   {Plugin::Settings::LayerType::eWMS, "wms"},
   {Plugin::Settings::LayerType::eWMTS, "wmts"},
});

// clang-format on

////////////////////////////////////////////////////////////////////////////////////////////////////

void from_json(nlohmann::json const& j, Plugin::Settings::Map& o) {
  cs::core::Settings::deserialize(j, "projection", o.mProjection);
  cs::core::Settings::deserialize(j, "type", o.mType);
  cs::core::Settings::deserialize(j, "url", o.mURL);
  cs::core::Settings::deserialize(j, "config", o.mConfig);

  if (o.mType == Plugin::Settings::LayerType::eNone) {
    throw cs::core::Settings::DeserializationException(
        "'type'", "Invalid layer type given! Should be either 'wms' or 'wmts'");
  }

  if (o.mProjection == Plugin::Settings::ProjectionType::eNone) {
    throw cs::core::Settings::DeserializationException("'projection'",
        "Invalid projection type given! Should be either 'mercator' or 'equirectangular'.");
  }
}

void to_json(nlohmann::json& j, Plugin::Settings::Map const& o) {
  cs::core::Settings::serialize(j, "projection", o.mProjection);
  cs::core::Settings::serialize(j, "type", o.mType);
  cs::core::Settings::serialize(j, "url", o.mURL);
  cs::core::Settings::serialize(j, "config", o.mConfig);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void from_json(nlohmann::json const& j, Plugin::Settings& o) {
  cs::core::Settings::deserialize(j, "defaultMap", o.mDefaultMap);
  cs::core::Settings::deserialize(j, "maps", o.mMaps);
}

void to_json(nlohmann::json& j, Plugin::Settings const& o) {
  cs::core::Settings::serialize(j, "defaultMap", o.mDefaultMap);
  cs::core::Settings::serialize(j, "maps", o.mMaps);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::init() {

  logger().info("Loading plugin...");

  mOnLoadConnection = mAllSettings->onLoad().connect([this]() { onLoad(); });
  mOnSaveConnection = mAllSettings->onSave().connect([this]() { onSave(); });

  // Add resources to gui.
  mGuiManager->executeJavascriptFile("../share/resources/gui/third-party/js/leaflet.js");
  mGuiManager->executeJavascriptFile(
      "../share/resources/gui/third-party/js/leaflet.markercluster.js");
  mGuiManager->addCSS("third-party/css/leaflet.css");

  mGuiManager->addCSS("css/csp-minimap.css");
  mGuiManager->addTemplate("minimap", "../share/resources/gui/csp-minimap-template.html");
  mGuiManager->executeJavascriptFile("../share/resources/gui/js/csp-minimap.js");

  // Register a callback to toggle the minimap.
  std::string callback = "minimap.toggle";
  mGuiManager->getGui()->registerCallback(callback, "Toggles the Minimap.", std::function([this]() {
    mGuiManager->getGui()->executeJavascript(
        "document.querySelector('#minimap').classList.toggle('visible')");
  }));

  // Add a timeline button to toggle the minimap.
  mGuiManager->addTimelineButton("Toggle Minimap", "map", callback);

  // Add newly created bookmarks.
  mOnBookmarkAddedConnection = mGuiManager->onBookmarkAdded().connect(
      [this](uint32_t bookmarkID, cs::core::Settings::Bookmark const& bookmark) {
        onAddBookmark(mSolarSystem->pActiveObject.get(), bookmarkID, bookmark);
      });

  // Remove deleted bookmarks.
  mOnBookmarkRemovedConnection = mGuiManager->onBookmarkRemoved().connect(
      [this](uint32_t bookmarkID, cs::core::Settings::Bookmark const& /*bookmark*/) {
        mGuiManager->getGui()->callJavascript("CosmoScout.minimap.removeBookmark", bookmarkID);
      });

  // Update bookmarks and map layers if active body changes.
  mActiveObjectConnection = mSolarSystem->pActiveObject.connectAndTouch(
      [this](std::shared_ptr<const cs::scene::CelestialObject> const& body) {
        // First remove all bookmarks.
        mGuiManager->getGui()->callJavascript("CosmoScout.minimap.removeBookmarks");
        mGuiManager->getGui()->callJavascript("CosmoScout.minimap.configure", "");

        if (body) {
          // Add all layers configured for this body.
          auto mapSettings = mPluginSettings.mMaps.find(body->getCenterName());
          if (mapSettings != mPluginSettings.mMaps.end()) {
            nlohmann::json json = mapSettings->second;
            mGuiManager->getGui()->callJavascript("CosmoScout.minimap.configure", json.dump());
          } else if (mPluginSettings.mDefaultMap.has_value()) {
            nlohmann::json json = mPluginSettings.mDefaultMap.value();
            mGuiManager->getGui()->callJavascript("CosmoScout.minimap.configure", json.dump());
          }

          // Add all bookmarks with positions for this body.
          for (auto const& [id, bookmark] : mGuiManager->getBookmarks()) {
            onAddBookmark(body, id, bookmark);
          }
        }
      });

  // Load initial settings.
  onLoad();

  logger().info("Loading done.");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::deInit() {
  logger().info("Unloading plugin...");

  // Save settings as this plugin may get reloaded.
  onSave();

  mAllSettings->onLoad().disconnect(mOnLoadConnection);
  mAllSettings->onSave().disconnect(mOnSaveConnection);

  mGuiManager->onBookmarkAdded().disconnect(mOnBookmarkAddedConnection);
  mGuiManager->onBookmarkRemoved().disconnect(mOnBookmarkRemovedConnection);

  mGuiManager->removeTemplate("minimap");
  mGuiManager->removeCSS("css/csp-minimap.css");

  mGuiManager->removeTimelineButton("Toggle Minimap");
  mGuiManager->getGui()->unregisterCallback("minimap.toggle");

  mSolarSystem->pActiveObject.disconnect(mActiveObjectConnection);

  logger().info("Unloading done.");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::onAddBookmark(std::shared_ptr<const cs::scene::CelestialObject> const& activeObject,
    uint32_t bookmarkID, cs::core::Settings::Bookmark const& bookmark) {

  // Add only if it has a location and matches the currently active body.
  if (bookmark.mLocation && bookmark.mLocation.value().mPosition) {
    if (activeObject && activeObject->getCenterName() == bookmark.mLocation.value().mCenter) {
      auto radii = activeObject->getRadii();
      auto p     = cs::utils::convert::cartesianToLngLat(
          bookmark.mLocation.value().mPosition.value(), radii);
      p      = cs::utils::convert::toDegrees(p);
      auto c = bookmark.mColor.value_or(glm::vec3(0.8F, 0.8F, 1.0F)) * 255.F;
      mGuiManager->getGui()->callJavascript("CosmoScout.minimap.addBookmark", bookmarkID,
          fmt::format("rgb({}, {}, {})", c.r, c.g, c.b), p.x, p.y);
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::onLoad() {
  // Read settings from JSON.
  from_json(mAllSettings->mPlugins.at("csp-minimap"), mPluginSettings);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::onSave() {
  mAllSettings->mPlugins["csp-minimap"] = mPluginSettings;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::minimap
