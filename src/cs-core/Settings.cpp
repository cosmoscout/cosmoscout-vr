////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "Settings.hpp"

#include "../cs-scene/CelestialBody.hpp"
#include "../cs-utils/convert.hpp"
#include "SolarSystem.hpp"
#include "logger.hpp"

#include <fstream>
#include <iostream>

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
  Settings::deserialize(j, "existence", o.mExistence);
  Settings::deserialize(j, "radii", o.mRadii);
  Settings::deserialize(j, "position", o.mPosition);
  Settings::deserialize(j, "rotation", o.mRotation);
  Settings::deserialize(j, "scale", o.mScale);
  Settings::deserialize(j, "trackable", o.mTrackable);
}

void to_json(nlohmann::json& j, Settings::Anchor const& o) {
  Settings::serialize(j, "center", o.mCenter);
  Settings::serialize(j, "frame", o.mFrame);
  Settings::serialize(j, "existence", o.mExistence);
  Settings::serialize(j, "radii", o.mRadii);
  Settings::serialize(j, "position", o.mPosition);
  Settings::serialize(j, "rotation", o.mRotation);
  Settings::serialize(j, "scale", o.mScale);
  Settings::serialize(j, "trackable", o.mTrackable);
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
  Settings::deserialize(j, "center", o.pCenter);
  Settings::deserialize(j, "frame", o.pFrame);
  Settings::deserialize(j, "position", o.pPosition);
  Settings::deserialize(j, "rotation", o.pRotation);
}

void to_json(nlohmann::json& j, Settings::Observer const& o) {
  Settings::serialize(j, "center", o.pCenter);
  Settings::serialize(j, "frame", o.pFrame);
  Settings::serialize(j, "position", o.pPosition);
  Settings::serialize(j, "rotation", o.pRotation);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void from_json(nlohmann::json const& j, Settings::Bookmark::Location& o) {
  Settings::deserialize(j, "center", o.mCenter);
  Settings::deserialize(j, "frame", o.mFrame);
  Settings::deserialize(j, "position", o.mPosition);
  Settings::deserialize(j, "rotation", o.mRotation);
}

void to_json(nlohmann::json& j, Settings::Bookmark::Location const& o) {
  Settings::serialize(j, "center", o.mCenter);
  Settings::serialize(j, "frame", o.mFrame);
  Settings::serialize(j, "position", o.mPosition);
  Settings::serialize(j, "rotation", o.mRotation);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void from_json(nlohmann::json const& j, Settings::Bookmark::Time& o) {
  Settings::deserialize(j, "start", o.mStart);
  Settings::deserialize(j, "end", o.mEnd);
}

void to_json(nlohmann::json& j, Settings::Bookmark::Time const& o) {
  Settings::serialize(j, "start", o.mStart);
  Settings::serialize(j, "end", o.mEnd);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void from_json(nlohmann::json const& j, Settings::Bookmark& o) {
  Settings::deserialize(j, "name", o.mName);
  Settings::deserialize(j, "description", o.mDescription);
  Settings::deserialize(j, "icon", o.mIcon);
  Settings::deserialize(j, "color", o.mColor);
  Settings::deserialize(j, "location", o.mLocation);
  Settings::deserialize(j, "time", o.mTime);
}

void to_json(nlohmann::json& j, Settings::Bookmark const& o) {
  Settings::serialize(j, "name", o.mName);
  Settings::serialize(j, "description", o.mDescription);
  Settings::serialize(j, "icon", o.mIcon);
  Settings::serialize(j, "color", o.mColor);
  Settings::serialize(j, "location", o.mLocation);
  Settings::serialize(j, "time", o.mTime);
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

void from_json(nlohmann::json const& j, Settings::Graphics& o) {
  Settings::deserialize(j, "enableVsync", o.pEnableVsync);
  Settings::deserialize(j, "worldUIScale", o.pWorldUIScale);
  Settings::deserialize(j, "mainUIScale", o.pMainUIScale);
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
  Settings::deserialize(j, "fixedSunDirection", o.pFixedSunDirection);
}

void to_json(nlohmann::json& j, Settings::Graphics const& o) {
  Settings::serialize(j, "enableVsync", o.pEnableVsync);
  Settings::serialize(j, "worldUIScale", o.pWorldUIScale);
  Settings::serialize(j, "mainUIScale", o.pMainUIScale);
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
  Settings::serialize(j, "fixedSunDirection", o.pFixedSunDirection);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void from_json(nlohmann::json const& j, Settings& o) {
  Settings::deserialize(j, "startDate", o.mStartDate);
  Settings::deserialize(j, "resetDate", o.mResetDate);
  Settings::deserialize(j, "observer", o.mObserver);
  Settings::deserialize(j, "spiceKernel", o.pSpiceKernel);
  Settings::deserialize(j, "sceneScale", o.mSceneScale);
  Settings::deserialize(j, "guiPosition", o.mGuiPosition);
  Settings::deserialize(j, "graphics", o.mGraphics);
  Settings::deserialize(j, "enableUserInterface", o.pEnableUserInterface);
  Settings::deserialize(j, "enableMouseRay", o.pEnableMouseRay);
  Settings::deserialize(j, "enableSensorSizeControl", o.pEnableSensorSizeControl);
  Settings::deserialize(j, "logLevelFile", o.pLogLevelFile);
  Settings::deserialize(j, "logLevelConsole", o.pLogLevelConsole);
  Settings::deserialize(j, "logLevelScreen", o.pLogLevelScreen);
  Settings::deserialize(j, "anchors", o.mAnchors);
  Settings::deserialize(j, "plugins", o.mPlugins);
  Settings::deserialize(j, "minDate", o.pMinDate);
  Settings::deserialize(j, "maxDate", o.pMaxDate);
  Settings::deserialize(j, "downloadData", o.mDownloadData);
  Settings::deserialize(j, "bookmarks", o.mBookmarks);
  Settings::deserialize(j, "commandHistory", o.mCommandHistory);
}

void to_json(nlohmann::json& j, Settings const& o) {
  Settings::serialize(j, "startDate", o.mStartDate);
  Settings::serialize(j, "resetDate", o.mResetDate);
  Settings::serialize(j, "observer", o.mObserver);
  Settings::serialize(j, "spiceKernel", o.pSpiceKernel);
  Settings::serialize(j, "sceneScale", o.mSceneScale);
  Settings::serialize(j, "guiPosition", o.mGuiPosition);
  Settings::serialize(j, "graphics", o.mGraphics);
  Settings::serialize(j, "enableUserInterface", o.pEnableUserInterface);
  Settings::serialize(j, "enableMouseRay", o.pEnableMouseRay);
  Settings::serialize(j, "enableSensorSizeControl", o.pEnableSensorSizeControl);
  Settings::serialize(j, "logLevelFile", o.pLogLevelFile);
  Settings::serialize(j, "logLevelConsole", o.pLogLevelConsole);
  Settings::serialize(j, "logLevelScreen", o.pLogLevelScreen);
  Settings::serialize(j, "anchors", o.mAnchors);
  Settings::serialize(j, "plugins", o.mPlugins);
  Settings::serialize(j, "minDate", o.pMinDate);
  Settings::serialize(j, "maxDate", o.pMaxDate);
  Settings::serialize(j, "downloadData", o.mDownloadData);
  Settings::serialize(j, "bookmarks", o.mBookmarks);
  Settings::serialize(j, "commandHistory", o.mCommandHistory);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

utils::Signal<> const& Settings::onLoad() const {
  return mOnLoad;
}

utils::Signal<> const& Settings::onSave() const {
  return mOnSave;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Settings::loadFromFile(std::string const& fileName) {
  std::ifstream i(fileName);

  if (!i) {
    throw std::runtime_error("Cannot open file: '" + fileName + "'!");
  }

  nlohmann::json settings;
  i >> settings;

  from_json(settings, *this);

  // Notify listeners that values might have changed.
  mOnLoad.emit();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Settings::loadFromJson(std::string const& json) {

  nlohmann::json settings = nlohmann::json::parse(json);
  from_json(settings, *this);

  // Notify listeners that values might have changed.
  mOnLoad.emit();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Settings::saveToFile(std::string const& fileName) const {
  // Tell listeners that the settings are about to be saved.
  mOnSave.emit();

  // Write to a temporary file first.
  std::ofstream o(fileName + ".tmp");

  if (!o) {
    throw std::runtime_error("Cannot open file: '" + fileName + "'!");
  }

  nlohmann::json settings = *this;
  o << std::setw(2) << settings;

  o.close();

  // Remove the existing file (if any).
  std::remove(fileName.c_str());

  // All done, so we're safe to rename the file.
  std::rename((fileName + ".tmp").c_str(), fileName.c_str());
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::string Settings::saveToJson() const {
  // Tell listeners that the settings are about to be saved.
  mOnSave.emit();

  nlohmann::json settings = *this;

  // Use an indentation of two space.
  std::ostringstream o;
  o << std::setw(2) << settings;

  return o.str();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Settings::initAnchor(scene::CelestialAnchor& anchor, std::string const& anchorName) const {
  anchor.setCenterName(getAnchorCenter(anchorName));
  anchor.setFrameName(getAnchorFrame(anchorName));
  anchor.setAnchorPosition(getAnchorPosition(anchorName));
  anchor.setAnchorRotation(getAnchorRotation(anchorName));
  anchor.setAnchorScale(getAnchorScale(anchorName));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Settings::initAnchor(scene::CelestialObject& object, std::string const& anchorName) const {
  initAnchor(static_cast<scene::CelestialAnchor&>(object), anchorName);
  object.setRadii(getAnchorRadii(anchorName));
  object.setExistence(getAnchorExistence(anchorName));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Settings::initAnchor(scene::CelestialBody& body, std::string const& anchorName) const {
  initAnchor(static_cast<scene::CelestialObject&>(body), anchorName);
  body.pTrackable = getAnchorIsTrackable(anchorName);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

glm::dvec2 Settings::getAnchorExistence(std::string const& anchorName) const {
  auto anchor = mAnchors.find(anchorName);
  if (anchor == mAnchors.end()) {
    throw std::runtime_error("Failed to parse the 'existence' property of the anchor '" +
                             anchorName + "': No anchor with this name found in the settings!");
  }

  try {
    return glm::dvec2(utils::convert::time::toSpice(anchor->second.mExistence[0]),
        utils::convert::time::toSpice(anchor->second.mExistence[1]));
  } catch (std::exception const&) {
    throw std::runtime_error(
        "Failed to parse the 'existence' property of the anchor '" + anchorName +
        "': The dates should be given in the format: 1969-07-20T20:17:40.000Z");
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

glm::dvec3 Settings::getAnchorRadii(std::string const& anchorName) const {
  auto anchor = mAnchors.find(anchorName);
  if (anchor == mAnchors.end()) {
    throw std::runtime_error("Failed to read the 'radii' property of the anchor '" + anchorName +
                             "': No anchor with this name found in the settings!");
  }

  return anchor->second.mRadii.value_or(SolarSystem::getRadii(anchor->second.mCenter));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::string Settings::getAnchorCenter(std::string const& anchorName) const {
  auto anchor = mAnchors.find(anchorName);
  if (anchor == mAnchors.end()) {
    throw std::runtime_error("Failed to get the 'center' property of the anchor '" + anchorName +
                             "': No anchor with this name found in the settings!");
  }

  return anchor->second.mCenter;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::string Settings::getAnchorFrame(std::string const& anchorName) const {
  auto anchor = mAnchors.find(anchorName);
  if (anchor == mAnchors.end()) {
    throw std::runtime_error("Failed to get the 'frame' property of the anchor '" + anchorName +
                             "': No anchor with this name found in the settings!");
  }

  return anchor->second.mFrame;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

glm::dvec3 Settings::getAnchorPosition(std::string const& anchorName) const {
  auto anchor = mAnchors.find(anchorName);
  if (anchor == mAnchors.end()) {
    throw std::runtime_error("Failed to get the 'position' property of the anchor '" + anchorName +
                             "': No anchor with this name found in the settings!");
  }

  return anchor->second.mPosition.value_or(glm::dvec3(0.0));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

glm::dquat Settings::getAnchorRotation(std::string const& anchorName) const {
  auto anchor = mAnchors.find(anchorName);
  if (anchor == mAnchors.end()) {
    throw std::runtime_error("Failed to get the 'rotation' property of the anchor '" + anchorName +
                             "': No anchor with this name found in the settings!");
  }

  return anchor->second.mRotation.value_or(glm::dquat(1.0, 0.0, 0.0, 0.0));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

double Settings::getAnchorScale(std::string const& anchorName) const {
  auto anchor = mAnchors.find(anchorName);
  if (anchor == mAnchors.end()) {
    throw std::runtime_error("Failed to get the 'scale' property of the anchor '" + anchorName +
                             "': No anchor with this name found in the settings!");
  }

  return anchor->second.mScale.value_or(1.0);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool Settings::getAnchorIsTrackable(std::string const& anchorName) const {
  auto anchor = mAnchors.find(anchorName);
  if (anchor == mAnchors.end()) {
    throw std::runtime_error("Failed to get the 'trackable' property of the anchor '" + anchorName +
                             "': No anchor with this name found in the settings!");
  }

  return anchor->second.mTrackable.get();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

Settings::DeserializationException::DeserializationException(
    std::string property, std::string jsonError)
    : mProperty(std::move(property))
    , mJSONError(std::move(jsonError))
    , mMessage("While parsing property " + mProperty + ": " + mJSONError){};

////////////////////////////////////////////////////////////////////////////////////////////////////

const char* Settings::DeserializationException::what() const noexcept {
  return mMessage.c_str();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Settings::deserialize(
    nlohmann::json const& j, std::string const& property, nlohmann::json& target) {
  try {
    target = j.at(property);
  } catch (DeserializationException const& e) {
    throw DeserializationException(e.mProperty + " in '" + property + "'", e.mJSONError);
  } catch (std::exception const& e) {
    throw DeserializationException("'" + property + "'", e.what());
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cs::core
