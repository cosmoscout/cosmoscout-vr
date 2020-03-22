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

namespace nlohmann {

/// A template specialization for serialization and deserialization of spdlog::level::level_enum.
template <>
struct adl_serializer<spdlog::level::level_enum> {
  static void to_json(json& j, spdlog::level::level_enum opt) {
    j = spdlog::level::to_string_view(opt).data();
  }

  static void from_json(const json& j, spdlog::level::level_enum& opt) {
    opt = spdlog::level::from_str(j.get<std::string>());
  }
};

} // namespace nlohmann

namespace cs::core {

////////////////////////////////////////////////////////////////////////////////////////////////////

void from_json(nlohmann::json const& j, Settings::Anchor& o) {
  Settings::deserialize(j, "center", o.mCenter);
  Settings::deserialize(j, "frame", o.mFrame);
  Settings::deserialize(j, "startExistence", o.mStartExistence);
  Settings::deserialize(j, "endExistence", o.mEndExistence);
}

void to_json(nlohmann::json& j, Settings::Anchor const& o) {
  Settings::serialize(j, "center", o.mCenter);
  Settings::serialize(j, "frame", o.mFrame);
  Settings::serialize(j, "startExistence", o.mStartExistence);
  Settings::serialize(j, "endExistence", o.mEndExistence);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void from_json(nlohmann::json const& j, Settings::GuiPosition& o) {
  Settings::deserialize(j, "widthPixel", o.mWidthPixel);
  Settings::deserialize(j, "heightPixel", o.mHeightPixel);
  Settings::deserialize(j, "widthMeter", o.mWidthMeter);
  Settings::deserialize(j, "heightMeter", o.mHeightMeter);
  Settings::deserialize(j, "posXMeter", o.mPosXMeter);
  Settings::deserialize(j, "posYMeter", o.mPosYMeter);
  Settings::deserialize(j, "posZMeter", o.mPosZMeter);
  Settings::deserialize(j, "rotX", o.mRotX);
  Settings::deserialize(j, "rotY", o.mRotY);
  Settings::deserialize(j, "rotZ", o.mRotZ);
}

void to_json(nlohmann::json& j, Settings::GuiPosition const& o) {
  Settings::serialize(j, "widthPixel", o.mWidthPixel);
  Settings::serialize(j, "heightPixel", o.mHeightPixel);
  Settings::serialize(j, "widthMeter", o.mWidthMeter);
  Settings::serialize(j, "heightMeter", o.mHeightMeter);
  Settings::serialize(j, "posXMeter", o.mPosXMeter);
  Settings::serialize(j, "posYMeter", o.mPosYMeter);
  Settings::serialize(j, "posZMeter", o.mPosZMeter);
  Settings::serialize(j, "rotX", o.mRotX);
  Settings::serialize(j, "rotY", o.mRotY);
  Settings::serialize(j, "rotZ", o.mRotZ);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void from_json(nlohmann::json const& j, Settings::Observer& o) {
  Settings::deserialize(j, "center", o.mCenter);
  Settings::deserialize(j, "frame", o.mFrame);
  Settings::deserialize(j, "longitude", o.mLongitude);
  Settings::deserialize(j, "latitude", o.mLatitude);
  Settings::deserialize(j, "distance", o.mDistance);
}

void to_json(nlohmann::json& j, Settings::Observer const& o) {
  Settings::serialize(j, "center", o.mCenter);
  Settings::serialize(j, "frame", o.mFrame);
  Settings::serialize(j, "longitude", o.mLongitude);
  Settings::serialize(j, "latitude", o.mLatitude);
  Settings::serialize(j, "distance", o.mDistance);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void from_json(nlohmann::json const& j, Settings::Event::Location& o) {
  Settings::deserialize(j, "planet", o.mPlanet);
  Settings::deserialize(j, "place", o.mPlace);
}

void to_json(nlohmann::json& j, Settings::Event::Location const& o) {
  Settings::serialize(j, "planet", o.mPlanet);
  Settings::serialize(j, "place", o.mPlace);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void from_json(nlohmann::json const& j, Settings::Event& o) {
  Settings::deserialize(j, "start", o.mStart);
  Settings::deserialize(j, "content", o.mContent);
  Settings::deserialize(j, "id", o.mId);
  Settings::deserialize(j, "description", o.mDescription);
  Settings::deserialize(j, "style", o.mStyle);
  Settings::deserialize(j, "end", o.mEnd);
  Settings::deserialize(j, "location", o.mLocation);
}

void to_json(nlohmann::json& j, Settings::Event const& o) {
  Settings::serialize(j, "start", o.mStart);
  Settings::serialize(j, "content", o.mContent);
  Settings::serialize(j, "id", o.mId);
  Settings::serialize(j, "description", o.mDescription);
  Settings::serialize(j, "style", o.mStyle);
  Settings::serialize(j, "end", o.mEnd);
  Settings::serialize(j, "location", o.mLocation);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void from_json(nlohmann::json const& j, Settings::DownloadData& o) {
  Settings::deserialize(j, "url", o.mUrl);
  Settings::deserialize(j, "file", o.mFile);
}

void to_json(nlohmann::json& j, Settings::DownloadData const& o) {
  Settings::serialize(j, "url", o.mUrl);
  Settings::serialize(j, "file", o.mFile);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void from_json(nlohmann::json const& j, Settings::SceneScale& o) {
  Settings::deserialize(j, "minScale", o.mMinScale);
  Settings::deserialize(j, "maxScale", o.mMaxScale);
  Settings::deserialize(j, "closeVisualDistance", o.mCloseVisualDistance);
  Settings::deserialize(j, "farVisualDistance", o.mFarVisualDistance);
  Settings::deserialize(j, "closeRealDistance", o.mCloseRealDistance);
  Settings::deserialize(j, "farRealDistance", o.mFarRealDistance);
  Settings::deserialize(j, "lockWeight", o.mLockWeight);
  Settings::deserialize(j, "trackWeight", o.mTrackWeight);
  Settings::deserialize(j, "minObjectSize", o.mMinObjectSize);
  Settings::deserialize(j, "nearClip", o.mNearClip);
  Settings::deserialize(j, "minFarClip", o.mMinFarClip);
  Settings::deserialize(j, "maxFarClip", o.mMaxFarClip);
}

void to_json(nlohmann::json& j, Settings::SceneScale const& o) {
  Settings::serialize(j, "minScale", o.mMinScale);
  Settings::serialize(j, "maxScale", o.mMaxScale);
  Settings::serialize(j, "closeVisualDistance", o.mCloseVisualDistance);
  Settings::serialize(j, "farVisualDistance", o.mFarVisualDistance);
  Settings::serialize(j, "closeRealDistance", o.mCloseRealDistance);
  Settings::serialize(j, "farRealDistance", o.mFarRealDistance);
  Settings::serialize(j, "lockWeight", o.mLockWeight);
  Settings::serialize(j, "trackWeight", o.mTrackWeight);
  Settings::serialize(j, "minObjectSize", o.mMinObjectSize);
  Settings::serialize(j, "nearClip", o.mNearClip);
  Settings::serialize(j, "minFarClip", o.mMinFarClip);
  Settings::serialize(j, "maxFarClip", o.mMaxFarClip);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void from_json(nlohmann::json const& j, Settings::LogLevel& o) {
  Settings::deserialize(j, "file", o.mFile);
  Settings::deserialize(j, "console", o.mConsole);
  Settings::deserialize(j, "screen", o.mScreen);
}

void to_json(nlohmann::json& j, Settings::LogLevel const& o) {
  Settings::serialize(j, "file", o.mFile);
  Settings::serialize(j, "console", o.mConsole);
  Settings::serialize(j, "screen", o.mScreen);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void from_json(nlohmann::json const& j, Settings::Graphics& o) {
  Settings::deserialize(j, "widgetScale", o.pWidgetScale);
  Settings::deserialize(j, "heightScale", o.pHeightScale);
  Settings::deserialize(j, "enableHDR", o.pEnableHDR);
  Settings::deserialize(j, "enableLighting", o.pEnableLighting);
  Settings::deserialize(j, "lightingQuality", o.pLightingQuality);
  Settings::deserialize(j, "enableShadows", o.pEnableShadows);
  Settings::deserialize(j, "enableShadowsDebug", o.pEnableShadowsDebug);
  Settings::deserialize(j, "enableShadowsFreeze", o.pEnableShadowsFreeze);
  Settings::deserialize(j, "shadowMapResolution", o.pShadowMapResolution);
  Settings::deserialize(j, "shadowMapCascades", o.pShadowMapCascades);
  Settings::deserialize(j, "shadowMapBias", o.pShadowMapBias);
  Settings::deserialize(j, "shadowMapRange", o.pShadowMapRange);
  Settings::deserialize(j, "shadowMapExtension", o.pShadowMapExtension);
  Settings::deserialize(j, "shadowMapSplitDistribution", o.pShadowMapSplitDistribution);
  Settings::deserialize(j, "enableAutoExposure", o.pEnableAutoExposure);
  Settings::deserialize(j, "exposure", o.pExposure);
  Settings::deserialize(j, "autoExposureRange", o.pAutoExposureRange);
  Settings::deserialize(j, "exposureCompensation", o.pExposureCompensation);
  Settings::deserialize(j, "exposureAdaptionSpeed", o.pExposureAdaptionSpeed);
  Settings::deserialize(j, "sensorDiagonal", o.pSensorDiagonal);
  Settings::deserialize(j, "focalLength", o.pFocalLength);
  Settings::deserialize(j, "ambientBrightness", o.pAmbientBrightness);
  Settings::deserialize(j, "enableAutoGlow", o.pEnableAutoGlow);
  Settings::deserialize(j, "glowIntensity", o.pGlowIntensity);
}

void to_json(nlohmann::json& j, Settings::Graphics const& o) {
  Settings::serialize(j, "widgetScale", o.pWidgetScale);
  Settings::serialize(j, "heightScale", o.pHeightScale);
  Settings::serialize(j, "enableHDR", o.pEnableHDR);
  Settings::serialize(j, "enableLighting", o.pEnableLighting);
  Settings::serialize(j, "lightingQuality", o.pLightingQuality);
  Settings::serialize(j, "enableShadows", o.pEnableShadows);
  Settings::serialize(j, "enableShadowsDebug", o.pEnableShadowsDebug);
  Settings::serialize(j, "enableShadowsFreeze", o.pEnableShadowsFreeze);
  Settings::serialize(j, "shadowMapResolution", o.pShadowMapResolution);
  Settings::serialize(j, "shadowMapCascades", o.pShadowMapCascades);
  Settings::serialize(j, "shadowMapBias", o.pShadowMapBias);
  Settings::serialize(j, "shadowMapRange", o.pShadowMapRange);
  Settings::serialize(j, "shadowMapExtension", o.pShadowMapExtension);
  Settings::serialize(j, "shadowMapSplitDistribution", o.pShadowMapSplitDistribution);
  Settings::serialize(j, "enableAutoExposure", o.pEnableAutoExposure);
  Settings::serialize(j, "exposure", o.pExposure);
  Settings::serialize(j, "autoExposureRange", o.pAutoExposureRange);
  Settings::serialize(j, "exposureCompensation", o.pExposureCompensation);
  Settings::serialize(j, "exposureAdaptionSpeed", o.pExposureAdaptionSpeed);
  Settings::serialize(j, "sensorDiagonal", o.pSensorDiagonal);
  Settings::serialize(j, "focalLength", o.pFocalLength);
  Settings::serialize(j, "ambientBrightness", o.pAmbientBrightness);
  Settings::serialize(j, "enableAutoGlow", o.pEnableAutoGlow);
  Settings::serialize(j, "glowIntensity", o.pGlowIntensity);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void from_json(nlohmann::json const& j, Settings& o) {
  Settings::deserialize(j, "startDate", o.mStartDate);
  Settings::deserialize(j, "observer", o.mObserver);
  Settings::deserialize(j, "spiceKernel", o.mSpiceKernel);
  Settings::deserialize(j, "sceneScale", o.mSceneScale);
  Settings::deserialize(j, "guiPosition", o.mGuiPosition);
  Settings::deserialize(j, "graphics", o.mGraphics);
  Settings::deserialize(j, "enableMouseRay", o.pEnableMouseRay);
  Settings::deserialize(j, "enableSensorSizeControl", o.pEnableSensorSizeControl);
  Settings::deserialize(j, "logLevel", o.mLogLevel);
  Settings::deserialize(j, "anchors", o.mAnchors);
  Settings::deserialize(j, "plugins", o.mPlugins);
  Settings::deserialize(j, "startDate", o.mStartDate);
  Settings::deserialize(j, "minDate", o.mMinDate);
  Settings::deserialize(j, "maxDate", o.mMaxDate);
  Settings::deserialize(j, "downloadData", o.mDownloadData);
  Settings::deserialize(j, "events", o.mEvents);
}

void to_json(nlohmann::json& j, Settings const& o) {
  Settings::serialize(j, "startDate", o.mStartDate);
  Settings::serialize(j, "observer", o.mObserver);
  Settings::serialize(j, "spiceKernel", o.mSpiceKernel);
  Settings::serialize(j, "sceneScale", o.mSceneScale);
  Settings::serialize(j, "guiPosition", o.mGuiPosition);
  Settings::serialize(j, "graphics", o.mGraphics);
  Settings::serialize(j, "enableMouseRay", o.pEnableMouseRay);
  Settings::serialize(j, "enableSensorSizeControl", o.pEnableSensorSizeControl);
  Settings::serialize(j, "logLevel", o.mLogLevel);
  Settings::serialize(j, "anchors", o.mAnchors);
  Settings::serialize(j, "plugins", o.mPlugins);
  Settings::serialize(j, "startDate", o.mStartDate);
  Settings::serialize(j, "minDate", o.mMinDate);
  Settings::serialize(j, "maxDate", o.mMaxDate);
  Settings::serialize(j, "downloadData", o.mDownloadData);
  Settings::serialize(j, "events", o.mEvents);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Settings::read(std::string const& fileName) {
  std::ifstream  i(fileName);
  nlohmann::json settings;
  i >> settings;

  from_json(settings, *this);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Settings::write(std::string const& fileName) const {
  std::ofstream  o(fileName);
  nlohmann::json settings = *this;
  o << std::setw(2) << settings;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::pair<double, double> Settings::Anchor::getExistence() {
  std::pair<double, double> result;

  try {
    result.first =
        utils::convert::toSpiceTime(boost::posix_time::time_from_string(mStartExistence));
  } catch (std::exception const& e) {
    throw std::runtime_error("Failed to parse the 'startExistence' property of the anchor '" +
                             mCenter +
                             "'. The dates should be given in the format: 1969-07-20 20:17:40.000");
  }

  try {
    result.second = utils::convert::toSpiceTime(boost::posix_time::time_from_string(mEndExistence));
  } catch (std::exception const& e) {
    throw std::runtime_error("Failed to parse the 'endExistence' property of the anchor '" +
                             mCenter +
                             "'. The dates should be given in the format: 1969-07-20 20:17:40.000");
  }

  return result;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

Settings::DeserializationException::DeserializationException(
    std::string const& property, std::string const& jsonError)
    : mProperty(property)
    , mJSONError(jsonError)
    , mMessage("While parsing property " + mProperty + ": " + mJSONError){};

////////////////////////////////////////////////////////////////////////////////////////////////////

const char* Settings::DeserializationException::what() const noexcept {
  return mMessage.c_str();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cs::core
