////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "Model.hpp"

#include "../../logger.hpp"

#include "../../../src/cs-utils/filesystem.hpp"

namespace csp::atmospheres::models::cosmoscout {

////////////////////////////////////////////////////////////////////////////////////////////////////

void from_json(nlohmann::json const& j, Model::Settings& o) {
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

bool Model::init(
    nlohmann::json const& modelSettings, double planetRadius, double atmosphereRadius) {

  Settings settings;

  try {
    settings = modelSettings;
  } catch (std::exception const& e) {
    logger().error("Failed to parse atmosphere parameters: {}", e.what());
  }

  mShader.Destroy();

  auto sFrag = cs::utils::filesystem::loadToString(
      "../share/resources/shaders/csp-atmospheres/models/cosmoscout/model.glsl");

  cs::utils::replaceString(sFrag, "ATMO_RADIUS", cs::utils::toString(atmosphereRadius));
  cs::utils::replaceString(sFrag, "PLANET_RADIUS", cs::utils::toString(planetRadius));
  cs::utils::replaceString(
      sFrag, "ANISOTROPY_R", cs::utils::toString(settings.mRayleighAnisotropy));
  cs::utils::replaceString(sFrag, "ANISOTROPY_M", cs::utils::toString(settings.mMieAnisotropy));
  cs::utils::replaceString(sFrag, "HEIGHT_R", cs::utils::toString(settings.mRayleighHeight));
  cs::utils::replaceString(sFrag, "HEIGHT_M", cs::utils::toString(settings.mMieHeight));
  cs::utils::replaceString(sFrag, "BETA_R",
      fmt::format("vec3({}, {}, {})", settings.mRayleighScattering[0],
          settings.mRayleighScattering[1], settings.mRayleighScattering[2]));
  cs::utils::replaceString(sFrag, "BETA_M",
      fmt::format("vec3({}, {}, {})", settings.mMieScattering[0], settings.mMieScattering[1],
          settings.mMieScattering[2]));
  cs::utils::replaceString(
      sFrag, "PRIMARY_RAY_STEPS", cs::utils::toString(settings.mPrimaryRaySteps));
  cs::utils::replaceString(
      sFrag, "SECONDARY_RAY_STEPS", cs::utils::toString(settings.mSecondaryRaySteps));

  mShader.InitFragmentShaderFromString(sFrag);

  return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

GLuint Model::getShader() const {
  return mShader.GetFragmentShader();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

GLuint Model::setUniforms(GLuint /*program*/, GLuint startTextureUnit) const {
  return startTextureUnit;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::atmospheres::models::cosmoscout
