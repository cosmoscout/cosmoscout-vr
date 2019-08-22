////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "Settings.hpp"

#include <fstream>

namespace cs::core {

////////////////////////////////////////////////////////////////////////////////////////////////////

void from_json(const nlohmann::json& j, Settings::Anchor& o) {
  o.mCenter         = j.at("center").get<std::string>();
  o.mFrame          = j.at("frame").get<std::string>();
  o.mStartExistence = j.at("startExistence").get<std::string>();
  o.mEndExistence   = j.at("endExistence").get<std::string>();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void from_json(const nlohmann::json& j, Settings::Gui& o) {
  o.mWidthPixel  = j.at("widthPixel").get<uint32_t>();
  o.mHeightPixel = j.at("heightPixel").get<uint32_t>();
  o.mWidthMeter  = j.at("widthMeter").get<double>();
  o.mHeightMeter = j.at("heightMeter").get<double>();
  o.mPosXMeter   = j.at("posXMeter").get<double>();
  o.mPosYMeter   = j.at("posYMeter").get<double>();
  o.mPosZMeter   = j.at("posZMeter").get<double>();
  o.mRotX        = j.at("rotX").get<double>();
  o.mRotY        = j.at("rotY").get<double>();
  o.mRotZ        = j.at("rotZ").get<double>();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void from_json(const nlohmann::json& j, Settings::Observer& o) {
  o.mCenter    = j.at("center").get<std::string>();
  o.mFrame     = j.at("frame").get<std::string>();
  o.mLongitude = j.at("longitude").get<double>();
  o.mLatitude  = j.at("latitude").get<double>();
  o.mDistance  = j.at("distance").get<double>();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void from_json(const nlohmann::json& j, Settings::Event& o) {
  o.mStart    = j.at("start").get<std::string>();
  o.mContent     = j.at("content").get<std::string>();
  o.mId = j.at("id").get<std::string>();
  auto iter = j.find("end");
  if (iter != j.end()) {
    o.mEnd = iter->get<std::optional<std::string>>();
  }
  iter = j.find("style");
  if (iter != j.end()) {
    o.mStyle = iter->get<std::optional<std::string>>();
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void from_json(const nlohmann::json& j, Settings& o) {
  o.mStartDate   = j.at("startDate").get<std::string>();
  o.mMinDate     = j.at("minDate").get<std::string>();
  o.mMaxDate     = j.at("maxDate").get<std::string>();
  o.mObserver    = j.at("observer").get<Settings::Observer>();
  o.mSpiceKernel = j.at("spiceKernel").get<std::string>();

  auto iter = j.find("gui");
  if (iter != j.end()) {
    o.mGui = iter->get<std::optional<Settings::Gui>>();
  }

  o.mWidgetScale    = j.at("widgetScale").get<float>();
  o.mEnableMouseRay = j.at("enableMouseRay").get<bool>();
  o.mAnchors        = j.at("anchors").get<std::map<std::string, Settings::Anchor>>();
  o.mPlugins        = j.at("plugins").get<std::map<std::string, nlohmann::json>>();
  o.mEvents         = j.at("events").get<std::vector<Settings::Event>>();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

Settings Settings::read(std::string const& fileName) {
  std::ifstream  i(fileName);
  nlohmann::json settings;
  i >> settings;

  return settings;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cs::core
