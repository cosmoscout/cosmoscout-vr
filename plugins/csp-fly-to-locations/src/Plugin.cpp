////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "Plugin.hpp"

#include "../../../src/cs-core/GuiManager.hpp"
#include "../../../src/cs-core/SolarSystem.hpp"
#include "logger.hpp"

////////////////////////////////////////////////////////////////////////////////////////////////////

EXPORT_FN cs::core::PluginBase* create() {
  return new csp::flytolocations::Plugin;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

EXPORT_FN void destroy(cs::core::PluginBase* pluginBase) {
  delete pluginBase; // NOLINT(cppcoreguidelines-owning-memory)
}

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace csp::flytolocations {

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::init() {

  logger().info("Loading plugin...");

  mGuiManager->addHtmlToGui(
      "fly-to-locations", "../share/resources/gui/fly-to-locations-templates.html");

  mGuiManager->addScriptToGuiFromJS("../share/resources/gui/js/csp-fly-to-locations.js");
  mGuiManager->addCssToGui("css/csp-fly-to-locations.css");

  mGuiManager->addPluginTabToSideBarFromHTML(
      "Bookmarks", "place", "../share/resources/gui/fly-to-locations-tab.html");

  // Add newly created bookmarks.
  mOnBookmarkAddedConnection = mGuiManager->onBookmarkAdded().connect(
      [this](uint32_t bookmarkID, cs::core::Settings::Bookmark const& bookmark) {
        onAddBookmark(bookmarkID, bookmark);
      });

  // Remove deleted bookmarks.
  mOnBookmarkRemovedConnection = mGuiManager->onBookmarkRemoved().connect(
      [this](uint32_t bookmarkID, cs::core::Settings::Bookmark const& /*bookmark*/) {
        mGuiManager->getGui()->callJavascript(
            "CosmoScout.flyToLocations.removeBookmark", bookmarkID);
      });

  // Update bookmark-list if active body changes.
  mActiveBodyConnection = mSolarSystem->pActiveBody.connectAndTouch(
      [this](std::shared_ptr<cs::scene::CelestialBody> const& body) {
        mGuiManager->getGui()->callJavascript(
            "CosmoScout.gui.clearHtml", "flytolocations-bookmarks-list");

        // If no body is set, we are in free space.
        std::string center = body ? body->getCenterName() : "Solar System Barycenter";

        // Add all list-bookmarks for this body.
        for (auto const& [id, bookmark] : mGuiManager->getBookmarks()) {
          if (bookmark.mLocation && bookmark.mLocation.value().mPosition) {
            if (center == bookmark.mLocation.value().mCenter) {
              mGuiManager->getGui()->callJavascript("CosmoScout.flyToLocations.addListBookmark", id,
                  bookmark.mName, bookmark.mTime.has_value());
            }
          }
        }
      });

  // Add all initial bookmarks.
  for (auto const& bookmark : mGuiManager->getBookmarks()) {
    onAddBookmark(bookmark.first, bookmark.second);
  }

  logger().info("Loading done.");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::deInit() {
  logger().info("Unloading plugin...");

  mGuiManager->removePluginTab("Bookmarks");

  mGuiManager->onBookmarkAdded().disconnect(mOnBookmarkAddedConnection);
  mGuiManager->onBookmarkRemoved().disconnect(mOnBookmarkRemovedConnection);

  mGuiManager->getGui()->callJavascript("CosmoScout.gui.unregisterHtml", "fly-to-locations");
  mGuiManager->getGui()->callJavascript(
      "CosmoScout.gui.unregisterCss", "css/csp-fly-to-locations.css");

  mSolarSystem->pActiveBody.disconnect(mActiveBodyConnection);

  logger().info("Unloading done.");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::onAddBookmark(uint32_t bookmarkID, cs::core::Settings::Bookmark const& bookmark) {
  // We only show bookmarks with locations.
  if (bookmark.mLocation) {
    if (bookmark.mIcon && !bookmark.mIcon.value().empty()) {
      // Add as grid-bookmark if it has an icon.
      mGuiManager->getGui()->callJavascript("CosmoScout.flyToLocations.addGridBookmark", bookmarkID,
          bookmark.mName, bookmark.mIcon.value());
    } else {
      // Add all other bookmars to the list, if they are relevant for the current body.
      if (mSolarSystem->getObserver().getCenterName() == bookmark.mLocation.value().mCenter) {
        mGuiManager->getGui()->callJavascript("CosmoScout.flyToLocations.addListBookmark",
            bookmarkID, bookmark.mName, bookmark.mTime.has_value());
      }
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::flytolocations
