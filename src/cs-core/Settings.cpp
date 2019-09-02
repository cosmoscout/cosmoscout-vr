////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "Settings.hpp"

#include <fstream>
#include <iostream>

namespace cs::core {

void CS_CORE_EXPORT parseSettingsSection(
    std::string const& sectionName, const std::function<void()>& f) {
  try {
    f();
  } catch (SettingsSectionException const& s) {
    throw SettingsSectionException(sectionName + "." + s.sectionName, s.what());
  } catch (std::exception const& e) { throw SettingsSectionException(sectionName, e.what()); }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void from_json(const nlohmann::json& j, Settings::Anchor& o) {
  o.mCenter         = parseProperty<std::string>("center", j);
  o.mFrame          = parseProperty<std::string>("frame", j);
  o.mStartExistence = parseProperty<std::string>("startExistence", j);
  o.mEndExistence   = parseProperty<std::string>("endExistence", j);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void from_json(const nlohmann::json& j, Settings::Gui& o) {
  parseSettingsSection("gui", [&] {
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
  });
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void from_json(const nlohmann::json& j, Settings::Observer& o) {
  parseSettingsSection("observer", [&] {
    o.mCenter    = parseProperty<std::string>("center", j);
    o.mFrame     = parseProperty<std::string>("frame", j);
    o.mLongitude = parseProperty<double>("longitude", j);
    o.mLatitude  = parseProperty<double>("latitude", j);
    o.mDistance  = parseProperty<double>("distance", j);
  });
}

////////////////////////////////////////////////////////////////////////////////////////////////////

#include <iostream>

void from_json(const nlohmann::json& j, Settings& o) {
  o.mStartDate   = parseProperty<std::string>("startDate", j);
  o.mObserver    = j.at("observer").get<Settings::Observer>();
  o.mSpiceKernel = parseProperty<std::string>("spiceKernel", j);

  auto iter = j.find("gui");
  if (iter != j.end()) {
    o.mGui = iter->get<std::optional<Settings::Gui>>();
  }

  o.mWidgetScale    = parseProperty<float>("widgetScale", j);
  o.mEnableMouseRay = parseProperty<bool>("enableMouseRay", j);
  parseSettingsSection("anchors",
      [&] { o.mAnchors = j.at("anchors").get<std::map<std::string, Settings::Anchor>>(); });

  parseSettingsSection("plugins",
      [&] { o.mPlugins = j.at("plugins").get<std::map<std::string, nlohmann::json>>(); });
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
