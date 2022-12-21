////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "Model.hpp"

#include "../../../src/cs-utils/filesystem.hpp"

namespace csp::atmospheres::models::cosmoscout {

////////////////////////////////////////////////////////////////////////////////////////////////////

void from_json(nlohmann::json const& j, Model::Settings& o) {
  cs::core::Settings::deserialize(j, "radius", o.mRadius);
  cs::core::Settings::deserialize(j, "mieHeight", o.mMieHeight);
  cs::core::Settings::deserialize(j, "mieScattering", o.mMieScattering);
  cs::core::Settings::deserialize(j, "mieAnisotropy", o.mMieAnisotropy);
  cs::core::Settings::deserialize(j, "rayleighHeight", o.mRayleighHeight);
  cs::core::Settings::deserialize(j, "rayleighScattering", o.mRayleighScattering);
  cs::core::Settings::deserialize(j, "rayleighAnisotropy", o.mRayleighAnisotropy);
  cs::core::Settings::deserialize(j, "primaryRaySteps", o.mPrimaryRaySteps);
  cs::core::Settings::deserialize(j, "secondaryRaySteps", o.mSecondaryRaySteps);
}

void to_json(nlohmann::json& j, Model::Settings const& o) {
  cs::core::Settings::serialize(j, "radius", o.mRadius);
  cs::core::Settings::serialize(j, "mieHeight", o.mMieHeight);
  cs::core::Settings::serialize(j, "mieScattering", o.mMieScattering);
  cs::core::Settings::serialize(j, "mieAnisotropy", o.mMieAnisotropy);
  cs::core::Settings::serialize(j, "rayleighHeight", o.mRayleighHeight);
  cs::core::Settings::serialize(j, "rayleighScattering", o.mRayleighScattering);
  cs::core::Settings::serialize(j, "rayleighAnisotropy", o.mRayleighAnisotropy);
  cs::core::Settings::serialize(j, "primaryRaySteps", o.mPrimaryRaySteps);
  cs::core::Settings::serialize(j, "secondaryRaySteps", o.mSecondaryRaySteps);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool Model::init(nlohmann::json modelSettings, double planetRadius) {
  if (mPreviousSettings == modelSettings && mPlanetRadius == planetRadius) {
    return false;
  }

  mPreviousSettings = std::move(modelSettings);
  mSettings         = mPreviousSettings;
  mPlanetRadius     = planetRadius;

  mShader.Destroy();

  auto sFrag = cs::utils::filesystem::loadToString(
      "../share/resources/shaders/csp-atmospheres/models/cosmoscout/model.glsl");

  cs::utils::replaceString(sFrag, "ATMO_RADIUS", cs::utils::toString(mSettings.mRadius));
  cs::utils::replaceString(sFrag, "PLANET_RADIUS", cs::utils::toString(mPlanetRadius));
  cs::utils::replaceString(
      sFrag, "ANISOTROPY_R", cs::utils::toString(mSettings.mRayleighAnisotropy));
  cs::utils::replaceString(sFrag, "ANISOTROPY_M", cs::utils::toString(mSettings.mMieAnisotropy));
  cs::utils::replaceString(sFrag, "HEIGHT_R", cs::utils::toString(mSettings.mRayleighHeight));
  cs::utils::replaceString(sFrag, "HEIGHT_M", cs::utils::toString(mSettings.mMieHeight));
  cs::utils::replaceString(sFrag, "BETA_R",
      fmt::format("vec3({}, {}, {})", mSettings.mRayleighScattering[0],
          mSettings.mRayleighScattering[1], mSettings.mRayleighScattering[2]));
  cs::utils::replaceString(sFrag, "BETA_M",
      fmt::format("vec3({}, {}, {})", mSettings.mMieScattering[0], mSettings.mMieScattering[1],
          mSettings.mMieScattering[2]));
  cs::utils::replaceString(
      sFrag, "PRIMARY_RAY_STEPS", cs::utils::toString(mSettings.mPrimaryRaySteps));
  cs::utils::replaceString(
      sFrag, "SECONDARY_RAY_STEPS", cs::utils::toString(mSettings.mSecondaryRaySteps));

  mShader.InitFragmentShaderFromString(sFrag);

  return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

GLuint Model::getShader() const {
  return mShader.GetFragmentShader();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

GLuint Model::setUniforms(GLuint program, GLuint startTextureUnit) const {
  return startTextureUnit;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::atmospheres::models::cosmoscout
