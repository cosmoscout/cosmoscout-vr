////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "Settings.hpp"

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

namespace cs::utils {

////////////////////////////////////////////////////////////////////////////////////////////////////

void from_json(nlohmann::json const&                                               j,
    ObservableMap<std::string, std::shared_ptr<const cs::scene::CelestialObject>>& o) {

  o.clear();

  for (auto const& el : j.items()) {
    std::string                name = el.key();
    nlohmann::json             data = el.value();
    cs::scene::CelestialObject object;

    // First, we parse the required parameters.
    std::string                center, frame;
    std::array<std::string, 2> existence;
    cs::core::Settings::deserialize(data, "center", center);
    cs::core::Settings::deserialize(data, "frame", frame);
    cs::core::Settings::deserialize(data, "existence", existence);

    object.setCenterName(center);
    object.setFrameName(frame);
    object.setExistenceAsStrings(existence);

    // All others are optional.
    std::optional<glm::dvec3> position, radii;
    std::optional<glm::dquat> rotation;
    std::optional<double>     scale, bodyCullingRadius, orbitCullingRadius;
    std::optional<bool>       trackable, collidable;
    cs::core::Settings::deserialize(data, "position", position);
    cs::core::Settings::deserialize(data, "rotation", rotation);
    cs::core::Settings::deserialize(data, "scale", scale);
    cs::core::Settings::deserialize(data, "radii", radii);
    cs::core::Settings::deserialize(data, "bodyCullingRadius", bodyCullingRadius);
    cs::core::Settings::deserialize(data, "orbitCullingRadius", orbitCullingRadius);
    cs::core::Settings::deserialize(data, "trackable", trackable);
    cs::core::Settings::deserialize(data, "collidable", collidable);

    if (position.has_value()) {
      object.setPosition(position.value());
    }
    if (rotation.has_value()) {
      object.setRotation(rotation.value());
    }
    if (scale.has_value()) {
      object.setScale(scale.value());
    }
    if (radii.has_value()) {
      object.setRadii(radii.value());
    }
    if (bodyCullingRadius.has_value()) {
      object.setBodyCullingRadius(bodyCullingRadius.value());
    }
    if (orbitCullingRadius.has_value()) {
      object.setOrbitCullingRadius(orbitCullingRadius.value());
    }
    if (trackable.has_value()) {
      object.setIsTrackable(trackable.value());
    }
    if (collidable.has_value()) {
      object.setIsCollidable(collidable.value());
    }

    o.insert(name, std::make_shared<cs::scene::CelestialObject>(object));
  }
}

void to_json(nlohmann::json&                                                             j,
    ObservableMap<std::string, std::shared_ptr<const cs::scene::CelestialObject>> const& o) {

  j.clear();

  for (auto const& [name, object] : o) {

    nlohmann::json i;

    cs::core::Settings::serialize(i, "center", object->getCenterName());
    cs::core::Settings::serialize(i, "frame", object->getFrameName());
    cs::core::Settings::serialize(i, "existence", object->getExistenceAsStrings());

    if (object->getPosition() != glm::dvec3(0.0, 0.0, 0.0)) {
      cs::core::Settings::serialize(i, "position", object->getPosition());
    }
    if (object->getRotation() != glm::dquat(1.0, 0.0, 0.0, 0.0)) {
      cs::core::Settings::serialize(i, "rotation", object->getRotation());
    }
    if (object->getScale() != 1.0) {
      cs::core::Settings::serialize(i, "scale", object->getScale());
    }
    if (object->hasCustomRadii()) {
      cs::core::Settings::serialize(i, "radii", object->getRadii());
    }
    if (object->getBodyCullingRadius() != 0) {
      cs::core::Settings::serialize(i, "bodyCullingRadius", object->getBodyCullingRadius());
    }
    if (object->getOrbitCullingRadius() != 0) {
      cs::core::Settings::serialize(i, "orbitCullingRadius", object->getOrbitCullingRadius());
    }
    if (!object->getIsTrackable()) {
      cs::core::Settings::serialize(i, "trackable", object->getIsTrackable());
    }
    if (!object->getIsCollidable()) {
      cs::core::Settings::serialize(i, "collidable", object->getIsCollidable());
    }

    j[name] = i;
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cs::utils

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace cs::core {

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
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void from_json(nlohmann::json const& j, Settings::EclipseShadowMap& o) {
  Settings::deserialize(j, "texture", o.mTexture);
}

void to_json(nlohmann::json& j, Settings::EclipseShadowMap const& o) {
  Settings::serialize(j, "texture", o.mTexture);
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
  Settings::deserialize(j, "ambientOcclusion", o.pAmbientOcclusion);
  Settings::deserialize(j, "glareIntensity", o.pGlareIntensity);
  Settings::deserialize(j, "glareRadius", o.pGlareQuality);
  Settings::deserialize(j, "glareMode", o.pGlareMode);
  Settings::deserialize(j, "toneMappingMode", o.pToneMappingMode);
  Settings::deserialize(j, "enableBicubicGlareFiltering", o.pEnableBicubicGlareFilter);
  Settings::deserialize(j, "fixedSunDirection", o.pFixedSunDirection);
  Settings::deserialize(j, "eclipseShadowMaps", o.mEclipseShadowMaps);
  Settings::deserialize(j, "eclipseShadowMode", o.pEclipseShadowMode);
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
  Settings::serialize(j, "ambientOcclusion", o.pAmbientOcclusion);
  Settings::serialize(j, "glareIntensity", o.pGlareIntensity);
  Settings::serialize(j, "glareRadius", o.pGlareQuality);
  Settings::serialize(j, "glareMode", o.pGlareMode);
  Settings::serialize(j, "toneMappingMode", o.pToneMappingMode);
  Settings::serialize(j, "enableBicubicGlareFiltering", o.pEnableBicubicGlareFilter);
  Settings::serialize(j, "fixedSunDirection", o.pFixedSunDirection);
  Settings::serialize(j, "eclipseShadowMaps", o.mEclipseShadowMaps);
  Settings::serialize(j, "eclipseShadowMode", o.pEclipseShadowMode);
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
  Settings::deserialize(j, "logLevelGL", o.pLogLevelGL);
  Settings::deserialize(j, "objects", o.mObjects);
  Settings::deserialize(j, "plugins", o.mPlugins);
  Settings::deserialize(j, "minDate", o.pMinDate);
  Settings::deserialize(j, "maxDate", o.pMaxDate);
  Settings::deserialize(j, "timeSpeed", o.pTimeSpeed);
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
  Settings::serialize(j, "logLevelGL", o.pLogLevelGL);
  Settings::serialize(j, "objects", o.mObjects);
  Settings::serialize(j, "plugins", o.mPlugins);
  Settings::serialize(j, "minDate", o.pMinDate);
  Settings::serialize(j, "maxDate", o.pMaxDate);
  Settings::serialize(j, "timeSpeed", o.pTimeSpeed);
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
