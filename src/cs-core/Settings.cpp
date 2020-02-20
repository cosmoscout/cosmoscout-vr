////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "Settings.hpp"
#include "../cs-utils/convert.hpp"

#include <fstream>
#include <iostream>
#include <spdlog/spdlog.h>

namespace cs::core {

////////////////////////////////////////////////////////////////////////////////////////////////////

void from_json(const nlohmann::json& j, Settings::Anchor& o) {
  o.mCenter         = parseProperty<std::string>("center", j);
  o.mFrame          = parseProperty<std::string>("frame", j);
  o.mStartExistence = parseProperty<std::string>("startExistence", j);
  o.mEndExistence   = parseProperty<std::string>("endExistence", j);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void from_json(const nlohmann::json& j, Settings::Gui& o) {
  o.mWidthPixel  = parseProperty<uint32_t>("widthPixel", j);
  o.mHeightPixel = parseProperty<uint32_t>("heightPixel", j);
  o.mWidthMeter  = parseProperty<double>("widthMeter", j);
  o.mHeightMeter = parseProperty<double>("heightMeter", j);
  o.mPosXMeter   = parseProperty<double>("posXMeter", j);
  o.mPosYMeter   = parseProperty<double>("posYMeter", j);
  o.mPosZMeter   = parseProperty<double>("posZMeter", j);
  o.mRotX        = parseProperty<double>("rotX", j);
  o.mRotY        = parseProperty<double>("rotY", j);
  o.mRotZ        = parseProperty<double>("rotZ", j);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void from_json(const nlohmann::json& j, Settings::Observer& o) {
  o.mCenter    = parseProperty<std::string>("center", j);
  o.mFrame     = parseProperty<std::string>("frame", j);
  o.mLongitude = parseProperty<double>("longitude", j);
  o.mLatitude  = parseProperty<double>("latitude", j);
  o.mDistance  = parseProperty<double>("distance", j);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void from_json(const nlohmann::json& j, Settings::Location& o) {
  o.mPlanet = parseProperty<std::string>("planet", j);
  o.mPlace  = parseProperty<std::string>("place", j);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void from_json(const nlohmann::json& j, Settings::Event& o) {
  o.mStart       = parseProperty<std::string>("start", j);
  o.mContent     = parseProperty<std::string>("content", j);
  o.mStyle       = parseProperty<std::string>("style", j);
  o.mId          = parseProperty<std::string>("id", j);
  o.mEnd         = parseOptionalSection<std::string>("end", j);
  o.mDescription = parseProperty<std::string>("description", j);
  o.mLocation    = parseOptionalSection<Settings::Location>("location", j);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void from_json(const nlohmann::json& j, Settings::DownloadData& o) {
  o.mUrl  = parseProperty<std::string>("url", j);
  o.mFile = parseProperty<std::string>("file", j);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void from_json(const nlohmann::json& j, Settings::SceneScale& o) {
  o.mMinScale            = parseProperty<double>("minScale", j);
  o.mMaxScale            = parseProperty<double>("maxScale", j);
  o.mCloseVisualDistance = parseProperty<double>("closeVisualDistance", j);
  o.mFarVisualDistance   = parseProperty<double>("farVisualDistance", j);
  o.mCloseRealDistance   = parseProperty<double>("closeRealDistance", j);
  o.mFarRealDistance     = parseProperty<double>("farRealDistance", j);
  o.mLockWeight          = parseProperty<double>("lockWeight", j);
  o.mTrackWeight         = parseProperty<double>("trackWeight", j);
  o.mMinObjectSize       = parseProperty<double>("minObjectSize", j);
  o.mNearClip            = parseProperty<double>("nearClip", j);
  o.mMinFarClip          = parseProperty<double>("minFarClip", j);
  o.mMaxFarClip          = parseProperty<double>("maxFarClip", j);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void from_json(const nlohmann::json& j, Settings& o) {
  o.mStartDate      = parseProperty<std::string>("startDate", j);
  o.mObserver       = parseSection<Settings::Observer>("observer", j);
  o.mSpiceKernel    = parseProperty<std::string>("spiceKernel", j);
  o.mSceneScale     = parseProperty<Settings::SceneScale>("sceneScale", j);
  o.mGui            = parseOptionalSection<Settings::Gui>("gui", j);
  o.mWidgetScale    = parseProperty<float>("widgetScale", j);
  o.mEnableMouseRay = parseProperty<bool>("enableMouseRay", j);
  o.mAnchors        = parseMap<std::string, Settings::Anchor>("anchors", j);
  o.mPlugins        = parseMap<std::string, nlohmann::json>("plugins", j);
  o.mStartDate      = parseProperty<std::string>("startDate", j);
  o.mMinDate        = parseProperty<std::string>("minDate", j);
  o.mMaxDate        = parseProperty<std::string>("maxDate", j);

  auto iter = j.find("downloadData");
  if (iter != j.end()) {
    o.mDownloadData = parseVector<Settings::DownloadData>("downloadData", j);
  }

  o.mEvents = parseVector<Settings::Event>("events", j);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

Settings Settings::read(std::string const& fileName) {
  std::ifstream  i(fileName);
  nlohmann::json settings;
  i >> settings;

  return settings;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void parseSection(std::string const& sectionName, const std::function<void()>& f) {
  try {
    f();
  } catch (SettingsSectionException const& s) {
    throw SettingsSectionException(sectionName + "." + s.sectionName, s.message);
  } catch (std::exception const& e) { throw SettingsSectionException(sectionName, e.what()); }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::pair<double, double> getExistenceFromSettings(
    std::pair<std::string, Settings::Anchor> const& anchor) {
  std::pair<double, double> result;

  try {
    result.first = utils::convert::toSpiceTime(
        boost::posix_time::time_from_string(anchor.second.mStartExistence));
  } catch (std::exception const& e) {
    throw std::runtime_error("Failed to parse the 'startExistence' property of the anchor '" +
                             anchor.first +
                             "'. The dates should be given in the format: 1969-07-20 20:17:40.000");
  }

  try {
    result.second = utils::convert::toSpiceTime(
        boost::posix_time::time_from_string(anchor.second.mEndExistence));
  } catch (std::exception const& e) {
    throw std::runtime_error("Failed to parse the 'endExistence' property of the anchor '" +
                             anchor.first +
                             "'. The dates should be given in the format: 1969-07-20 20:17:40.000");
  }

  return result;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cs::core
