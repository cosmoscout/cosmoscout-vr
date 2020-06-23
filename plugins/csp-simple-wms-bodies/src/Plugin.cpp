////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "Plugin.hpp"

#include "../../../src/cs-core/GuiManager.hpp"
#include "../../../src/cs-core/InputManager.hpp"
#include "../../../src/cs-core/Settings.hpp"
#include "../../../src/cs-core/SolarSystem.hpp"
#include "../../../src/cs-core/TimeControl.hpp"
#include "../../../src/cs-utils/logger.hpp"
#include "SimpleWMSBody.hpp"
#include "logger.hpp"

////////////////////////////////////////////////////////////////////////////////////////////////////

EXPORT_FN cs::core::PluginBase* create() {
  return new csp::simplewmsbodies::Plugin;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

EXPORT_FN void destroy(cs::core::PluginBase* pluginBase) {
  delete pluginBase; // NOLINT(cppcoreguidelines-owning-memory)
}

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace csp::simplewmsbodies {

////////////////////////////////////////////////////////////////////////////////////////////////////

void from_json(nlohmann::json const& j, Plugin::Settings::WMSConfig& o) {
  cs::core::Settings::deserialize(j, "copyright", o.mCopyright);
  cs::core::Settings::deserialize(j, "url", o.mUrl);
  cs::core::Settings::deserialize(j, "width", o.mWidth);
  cs::core::Settings::deserialize(j, "height", o.mHeight);
  cs::core::Settings::deserialize(j, "time", o.mTime);
  cs::core::Settings::deserialize(j, "preFetch", o.mPrefetchCount);
  cs::core::Settings::deserialize(j, "layers", o.mLayers);
  cs::core::Settings::deserialize(j, "timeSpan", o.mTimespan);
}

void to_json(nlohmann::json& j, Plugin::Settings::WMSConfig const& o) {
  cs::core::Settings::serialize(j, "copyright", o.mCopyright);
  cs::core::Settings::serialize(j, "url", o.mUrl);
  cs::core::Settings::serialize(j, "width", o.mWidth);
  cs::core::Settings::serialize(j, "height", o.mHeight);
  cs::core::Settings::serialize(j, "time", o.mTime);
  cs::core::Settings::serialize(j, "preFetch", o.mPrefetchCount);
  cs::core::Settings::serialize(j, "layers", o.mLayers);
  cs::core::Settings::serialize(j, "timeSpan", o.mTimespan);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void from_json(nlohmann::json const& j, Plugin::Settings::SimpleWMSBody& o) {
  cs::core::Settings::deserialize(j, "gridResolutionX", o.mGridResolutionX);
  cs::core::Settings::deserialize(j, "gridResolutionY", o.mGridResolutionY);
  cs::core::Settings::deserialize(j, "texture", o.mTexture);
  cs::core::Settings::deserialize(j, "activeWms", o.mActiveWMS);
  cs::core::Settings::deserialize(j, "wms", o.mWMS);
}

void to_json(nlohmann::json& j, Plugin::Settings::SimpleWMSBody const& o) {
  cs::core::Settings::serialize(j, "gridResolutionX", o.mGridResolutionX);
  cs::core::Settings::serialize(j, "gridResolutionY", o.mGridResolutionY);
  cs::core::Settings::serialize(j, "texture", o.mTexture);
  cs::core::Settings::serialize(j, "activeWms", o.mActiveWMS);
  cs::core::Settings::serialize(j, "wms", o.mWMS);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void from_json(nlohmann::json const& j, Plugin::Settings& o) {
  cs::core::Settings::deserialize(j, "mapCache", o.mMapCache);
  cs::core::Settings::deserialize(j, "bodies", o.mBodies);
}

void to_json(nlohmann::json& j, Plugin::Settings const& o) {
  cs::core::Settings::serialize(j, "mapCache", o.mMapCache);
  cs::core::Settings::serialize(j, "bodies", o.mBodies);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::init() {

  logger().info("Loading plugin...");

  mOnLoadConnection = mAllSettings->onLoad().connect([this]() { onLoad(); });

  mOnSaveConnection = mAllSettings->onSave().connect(
      [this]() { mAllSettings->mPlugins["csp-simple-wms-bodies"] = *mPluginSettings; });

  mGuiManager->addPluginTabToSideBarFromHTML(
      "WMS", "panorama", "../share/resources/gui/wms_body_tab.html");
  mGuiManager->addSettingsSectionToSideBarFromHTML(
      "WMS", "panorama", "../share/resources/gui/wms_settings.html");
  mGuiManager->addScriptToGuiFromJS("../share/resources/gui/js/csp-simple-wms-bodies.js");

  // Set whether to interpolate textures between timesteps (does not work when pre-fetch is
  // inactive).
  mGuiManager->getGui()->registerCallback("simpleWMSBodies.setEnableTimeInterpolation",
      "Enables or disables interpolation.",
      std::function([this](bool enable) { mPluginSettings->mEnableInterpolation = enable; }));

  // Set whether to display timespan.
  mGuiManager->getGui()->registerCallback("simpleWMSBodies.setEnableTimeSpan",
      "Enables or disables timespan.",
      std::function([this](bool enable) { mPluginSettings->mEnableTimespan = enable; }));

  // Set WMS source.
  mGuiManager->getGui()->registerCallback("simpleWMSBodies.setWMS",
      "Set the current planet's WMS source to the one with the given name.",
      std::function([this](std::string&& name) {
        auto body = std::dynamic_pointer_cast<SimpleWMSBody>(mSolarSystem->pActiveBody.get());
        if (body) {
          setWMSSource(body, name);

          // Replace bookmarks with timeintervals of the new WMS data set.
          // removeBookmarks();
          // addBookmarks(body->getTimeIntervals(), name, body->getCenterName(), body->getFrameName());
        }
      }));

  mActiveBodyConnection = mSolarSystem->pActiveBody.connectAndTouch(
      [this](std::shared_ptr<cs::scene::CelestialBody> const& body) {
        auto simpleWMSBody = std::dynamic_pointer_cast<SimpleWMSBody>(body);

        // Remove bookmarks from the old body.
        // removeBookmarks();

        mGuiManager->getGui()->callJavascript(
            "CosmoScout.sidebar.setTabEnabled", "WMS", simpleWMSBody != nullptr);

        if (!simpleWMSBody) {
          return;
        }

        mGuiManager->getGui()->callJavascript(
            "CosmoScout.gui.clearDropdown", "simpleWMSBodies.setWMS");
        mGuiManager->getGui()->callJavascript(
            "CosmoScout.gui.addDropdownValue", "simpleWMSBodies.setWMS", "None", "None", "false");

        auto const& settings = getBodySettings(simpleWMSBody);
        for (auto const& wms : settings.mWMS) {
          bool active = wms.first == settings.mActiveWMS;
          mGuiManager->getGui()->callJavascript("CosmoScout.gui.addDropdownValue",
              "simpleWMSBodies.setWMS", wms.first, wms.first, active);
          if (active) {
            mGuiManager->getGui()->callJavascript(
                "CosmoScout.simpleWMSBodies.setWMSDataCopyright", wms.second.mCopyright);

            // Only allow setting timespan if it is specified for the WMS data set.
            mGuiManager->getGui()->callJavascript(
                "CosmoScout.simpleWMSBodies.enableCheckBox", wms.second.mTimespan.value_or(false));
            if (!wms.second.mTimespan.value_or(false)) {
              mGuiManager->setCheckboxValue("simpleWMSBodies.setEnableTimeSpan", false);
              mPluginSettings->mEnableTimespan = false;
            }

            // Add bookmarks to timeline from the intervals of the active WMS.
            // addBookmarks(simpleWMSBody->getTimeIntervals(), wms.first,
            //     simpleWMSBody->getCenterName(), simpleWMSBody->getFrameName());
          }
        }
      });

  onLoad();

  logger().info("Loading done.");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::deInit() {

  logger().info("Unloading plugin...");

  for (auto const& simpleWMSBody : mSimpleWMSBodies) {
    mSolarSystem->unregisterBody(simpleWMSBody.second);
    mInputManager->unregisterSelectable(simpleWMSBody.second);
  }

  mSolarSystem->pActiveBody.disconnect(mActiveBodyConnection);

  mGuiManager->removePluginTab("WMS");
  mGuiManager->removeSettingsSection("WMS");
  
  mGuiManager->getGui()->callJavascript(
      "CosmoScout.gui.unregisterCss", "css/csp-simple-wms-bodies.css");

  mGuiManager->getGui()->unregisterCallback("simpleWMSBodies.setEnableTimeInterpolation");
  mGuiManager->getGui()->unregisterCallback("simpleWMSBodies.setEnableTimeSpan");
  mGuiManager->getGui()->unregisterCallback("simpleWMSBodies.setWMS");

  mAllSettings->onLoad().disconnect(mOnLoadConnection);
  mAllSettings->onSave().disconnect(mOnSaveConnection);

  logger().info("Unloading done.");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::onLoad() {

  // Read settings from JSON.
  from_json(mAllSettings->mPlugins.at("csp-simple-wms-bodies"), *mPluginSettings);

  // First try to re-configure existing simpleWMSBodies. We assume that they are similar if they
  // have the same name in the settings (which means they are attached to an anchor with the same
  // name).
  auto simpleWMSBody = mSimpleWMSBodies.begin();
  while (simpleWMSBody != mSimpleWMSBodies.end()) {
    auto settings = mPluginSettings->mBodies.find(simpleWMSBody->first);
    if (settings != mPluginSettings->mBodies.end()) {
      // If there are settings for this simpleWMSBody, reconfigure it.
      auto anchor                           = mAllSettings->mAnchors.find(settings->first);
      auto [tStartExistence, tEndExistence] = anchor->second.getExistence();
      simpleWMSBody->second->setStartExistence(tStartExistence);
      simpleWMSBody->second->setEndExistence(tEndExistence);
      simpleWMSBody->second->setFrameName(anchor->second.mFrame);
      simpleWMSBody->second->setCenterName(anchor->second.mCenter);
      simpleWMSBody->second->configure(settings->second);

      setWMSSource(simpleWMSBody->second, settings->second.mActiveWMS);

      // Add bookmarks to timeline from the intervals of the active WMS.
      // addBookmarks(simpleWMSBody->second->getTimeIntervals(), settings->second.mActiveWMS,
      //     simpleWMSBody->second->getCenterName(), simpleWMSBody->second->getFrameName());

      ++simpleWMSBody;
    } else {
      // Else delete it.
      mSolarSystem->unregisterBody(simpleWMSBody->second);
      mInputManager->unregisterSelectable(simpleWMSBody->second);
      simpleWMSBody = mSimpleWMSBodies.erase(simpleWMSBody);
    }
  }

  // Then add new simpleWMSBodies.
  for (auto const& settings : mPluginSettings->mBodies) {
    if (mSimpleWMSBodies.find(settings.first) != mSimpleWMSBodies.end()) {
      continue;
    }

    auto anchor = mAllSettings->mAnchors.find(settings.first);

    if (anchor == mAllSettings->mAnchors.end()) {
      throw std::runtime_error(
          "There is no Anchor \"" + settings.first + "\" defined in the settings.");
    }

    auto [tStartExistence, tEndExistence] = anchor->second.getExistence();

    auto simpleWMSBody =
        std::make_shared<SimpleWMSBody>(mAllSettings, mSolarSystem, mPluginSettings, mTimeControl,
            anchor->second.mCenter, anchor->second.mFrame, tStartExistence, tEndExistence);

    mSimpleWMSBodies.emplace(settings.first, simpleWMSBody);

    setWMSSource(simpleWMSBody, settings.second.mActiveWMS);
    simpleWMSBody->configure(settings.second);
    simpleWMSBody->setSun(mSolarSystem->getSun());

    // Add bookmarks to timeline from the intervals of the active WMS.
    // addBookmarks(simpleWMSBody->getTimeIntervals(), settings.second.mActiveWMS,
    //     simpleWMSBody->getCenterName(), simpleWMSBody->getFrameName());

    mSolarSystem->registerBody(simpleWMSBody);
    mInputManager->registerSelectable(simpleWMSBody);
  }

  mSolarSystem->pActiveBody.touch(mActiveBodyConnection);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::removeBookmarks() {
  for (auto const& id : mBookmarkIDs) {
    mGuiManager->removeBookmark(id);
  }
  mBookmarkIDs.clear();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::addBookmarks(std::vector<TimeInterval> timeIntervals, std::string wmsName,
    std::string planetName, std::string frameName) {
  for (auto const& interval : timeIntervals) {
    std::string start = utils::timeToString("%Y-%m-%d %H:%M:%S", interval.mStartTime);
    std::string end   = utils::timeToString("%Y-%m-%d %H:%M:%S", interval.mEndTime);

    cs::core::Settings::Bookmark::Location bookmarkLocation;
    bookmarkLocation.mCenter = planetName;
    bookmarkLocation.mFrame  = frameName;

    cs::core::Settings::Bookmark::Time bookmarkTime;
    bookmarkTime.mStart = start;
    if (start != end) {
      bookmarkTime.mEnd = end;
    }

    cs::core::Settings::Bookmark bookmark;
    bookmark.mName     = "WMS - " + wmsName;
    bookmark.mColor    = glm::vec3(0.6, 0.45, 0.7);
    bookmark.mLocation = bookmarkLocation;
    bookmark.mTime     = bookmarkTime;

    int bookmarkID = mGuiManager->addBookmark(bookmark);
    mBookmarkIDs.emplace_back(bookmarkID);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

Plugin::Settings::SimpleWMSBody& Plugin::getBodySettings(
    std::shared_ptr<SimpleWMSBody> const& simpleWMSBody) const {
  auto name = std::find_if(mSimpleWMSBodies.begin(), mSimpleWMSBodies.end(),
      [&](auto const& pair) { return pair.second == simpleWMSBody; });
  return mPluginSettings->mBodies.at(name->first);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::setWMSSource(
    std::shared_ptr<SimpleWMSBody> const& simpleWMSBody, std::string const& name) const {

  auto& settings = getBodySettings(simpleWMSBody);

  if (name == "None") {
    simpleWMSBody->setActiveWMS(nullptr);
    mGuiManager->getGui()->callJavascript("CosmoScout.simpleWMSBodies.setWMSDataCopyright", "");
    settings.mActiveWMS = "None";
  } else {
    auto dataset = settings.mWMS.find(name);
    if (dataset == settings.mWMS.end()) {
      logger().warn("Cannot set WMS dataset '{}': There is no dataset defined with this name! "
                    "Using first dataset instead...",
          name);
      dataset = settings.mWMS.begin();
    }

    settings.mActiveWMS = name;

    simpleWMSBody->setActiveWMS(std::make_shared<Plugin::Settings::WMSConfig>(dataset->second));

    mGuiManager->getGui()->callJavascript(
        "CosmoScout.simpleWMSBodies.setWMSDataCopyright", dataset->second.mCopyright);

    // Only allow setting timespan if it is specified for the WMS data set.
    mGuiManager->getGui()->callJavascript(
        "CosmoScout.simpleWMSBodies.enableCheckBox", dataset->second.mTimespan.value_or(false));
    if (!dataset->second.mTimespan.value_or(false)) {
      mGuiManager->setCheckboxValue("simpleWMSBodies.setEnableTimeSpan", false);
      mPluginSettings->mEnableTimespan = false;
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::simplewmsbodies
