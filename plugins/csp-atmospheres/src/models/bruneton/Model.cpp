////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-FileCopyrightText: 2017 Eric Bruneton
// SPDX-License-Identifier: MIT

#include "Model.hpp"

#include "../../logger.hpp"
#include "internal/CSVLoader.hpp"

#include <glm/gtc/constants.hpp>

namespace csp::atmospheres::models::bruneton {

////////////////////////////////////////////////////////////////////////////////////////////////////

void from_json(nlohmann::json const& j, Model::Settings::ScatteringComponent& o) {
  cs::core::Settings::deserialize(j, "betaSca", o.mBetaSca);
  cs::core::Settings::deserialize(j, "betaAbs", o.mBetaAbs);
  cs::core::Settings::deserialize(j, "phase", o.mPhase);
  cs::core::Settings::deserialize(j, "density", o.mDensity);
}

void to_json(nlohmann::json& j, Model::Settings::ScatteringComponent const& o) {
  cs::core::Settings::serialize(j, "betaSca", o.mBetaSca);
  cs::core::Settings::serialize(j, "betaAbs", o.mBetaAbs);
  cs::core::Settings::serialize(j, "phase", o.mPhase);
  cs::core::Settings::serialize(j, "density", o.mDensity);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void from_json(nlohmann::json const& j, Model::Settings::AbsorbingComponent& o) {
  cs::core::Settings::deserialize(j, "betaAbs", o.mBetaAbs);
  cs::core::Settings::deserialize(j, "density", o.mDensity);
}

void to_json(nlohmann::json& j, Model::Settings::AbsorbingComponent const& o) {
  cs::core::Settings::serialize(j, "betaAbs", o.mBetaAbs);
  cs::core::Settings::serialize(j, "density", o.mDensity);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void from_json(nlohmann::json const& j, Model::Settings& o) {
  cs::core::Settings::deserialize(j, "sunAngularRadius", o.mSunAngularRadius);
  cs::core::Settings::deserialize(j, "molecules", o.mMolecules);
  cs::core::Settings::deserialize(j, "aerosols", o.mAerosols);
  cs::core::Settings::deserialize(j, "ozone", o.mOzone);
  cs::core::Settings::deserialize(j, "groundAlbedo", o.mGroundAlbedo);
  cs::core::Settings::deserialize(j, "multiScatteringOrder", o.mMultiScatteringOrder);
  cs::core::Settings::deserialize(j, "sampleCountOpticalDepth", o.mSampleCountOpticalDepth);
  cs::core::Settings::deserialize(j, "sampleCountSingleScattering", o.mSampleCountSingleScattering);
  cs::core::Settings::deserialize(j, "sampleCountMultiScattering", o.mSampleCountMultiScattering);
  cs::core::Settings::deserialize(
      j, "sampleCountScatteringDensity", o.mSampleCountScatteringDensity);
  cs::core::Settings::deserialize(
      j, "sampleCountIndirectIrradiance", o.mSampleCountIndirectIrradiance);
  cs::core::Settings::deserialize(j, "transmittanceTextureWidth", o.mTransmittanceTextureWidth);
  cs::core::Settings::deserialize(j, "transmittanceTextureHeight", o.mTransmittanceTextureHeight);
  cs::core::Settings::deserialize(j, "scatteringTextureRSize", o.mScatteringTextureRSize);
  cs::core::Settings::deserialize(j, "scatteringTextureMuSize", o.mScatteringTextureMuSize);
  cs::core::Settings::deserialize(j, "scatteringTextureMuSSize", o.mScatteringTextureMuSSize);
  cs::core::Settings::deserialize(j, "scatteringTextureNuSize", o.mScatteringTextureNuSize);
  cs::core::Settings::deserialize(j, "irradianceTextureWidth", o.mIrradianceTextureWidth);
  cs::core::Settings::deserialize(j, "irradianceTextureHeight", o.mIrradianceTextureHeight);
}

void to_json(nlohmann::json& j, Model::Settings const& o) {
  cs::core::Settings::serialize(j, "sunAngularRadius", o.mSunAngularRadius);
  cs::core::Settings::serialize(j, "molecules", o.mMolecules);
  cs::core::Settings::serialize(j, "aerosols", o.mAerosols);
  cs::core::Settings::serialize(j, "ozone", o.mOzone);
  cs::core::Settings::serialize(j, "groundAlbedo", o.mGroundAlbedo);
  cs::core::Settings::serialize(j, "multiScatteringOrder", o.mMultiScatteringOrder);
  cs::core::Settings::serialize(j, "sampleCountOpticalDepth", o.mSampleCountOpticalDepth);
  cs::core::Settings::serialize(j, "sampleCountSingleScattering", o.mSampleCountSingleScattering);
  cs::core::Settings::serialize(j, "sampleCountMultiScattering", o.mSampleCountMultiScattering);
  cs::core::Settings::serialize(j, "sampleCountScatteringDensity", o.mSampleCountScatteringDensity);
  cs::core::Settings::serialize(
      j, "sampleCountIndirectIrradiance", o.mSampleCountIndirectIrradiance);
  cs::core::Settings::serialize(j, "transmittanceTextureWidth", o.mTransmittanceTextureWidth);
  cs::core::Settings::serialize(j, "transmittanceTextureHeight", o.mTransmittanceTextureHeight);
  cs::core::Settings::serialize(j, "scatteringTextureRSize", o.mScatteringTextureRSize);
  cs::core::Settings::serialize(j, "scatteringTextureMuSize", o.mScatteringTextureMuSize);
  cs::core::Settings::serialize(j, "scatteringTextureMuSSize", o.mScatteringTextureMuSSize);
  cs::core::Settings::serialize(j, "scatteringTextureNuSize", o.mScatteringTextureNuSize);
  cs::core::Settings::serialize(j, "irradianceTextureWidth", o.mIrradianceTextureWidth);
  cs::core::Settings::serialize(j, "irradianceTextureHeight", o.mIrradianceTextureHeight);
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

  internal::Params params;

  std::vector<double> wavelengths;
  uint32_t            densityCount = 0;

  params.mMolecules.mDensity =
      internal::CSVLoader::readDensity(settings.mMolecules.mDensity, densityCount);
  params.mMolecules.mPhase =
      internal::CSVLoader::readPhase(settings.mMolecules.mPhase, params.mWavelengths);
  params.mMolecules.mScattering =
      internal::CSVLoader::readExtinction(settings.mMolecules.mBetaSca, params.mWavelengths);
  params.mMolecules.mAbsorption =
      internal::CSVLoader::readExtinction(settings.mMolecules.mBetaAbs, params.mWavelengths);

  params.mAerosols.mDensity =
      internal::CSVLoader::readDensity(settings.mAerosols.mDensity, densityCount);
  params.mAerosols.mPhase =
      internal::CSVLoader::readPhase(settings.mAerosols.mPhase, params.mWavelengths);
  params.mAerosols.mScattering =
      internal::CSVLoader::readExtinction(settings.mAerosols.mBetaSca, params.mWavelengths);
  params.mAerosols.mAbsorption =
      internal::CSVLoader::readExtinction(settings.mAerosols.mBetaAbs, params.mWavelengths);

  if (settings.mOzone) {
    params.mOzone.mDensity =
        internal::CSVLoader::readDensity(settings.mOzone->mDensity, densityCount);
    params.mOzone.mAbsorption =
        internal::CSVLoader::readExtinction(settings.mOzone->mBetaAbs, params.mWavelengths);

  } else {
    params.mOzone.mDensity    = std::vector<double>(densityCount, 0.0);
    params.mOzone.mAbsorption = std::vector<double>(params.mWavelengths.size(), 0.0);
  }

  if (params.mWavelengths.size() < 3) {
    throw std::runtime_error(
        "At least three different wavelengths should be given in the scattering data!");
  } else if (params.mWavelengths.size() == 3 &&
             (params.mWavelengths[0] != internal::Implementation::kLambdaB ||
                 params.mWavelengths[1] != internal::Implementation::kLambdaG ||
                 params.mWavelengths[2] != internal::Implementation::kLambdaR)) {
    throw std::runtime_error("If three different wavelengths are given in the scattering data, "
                             "they should be exactly for 440 nm, 550 nm, and 680 nm!");
  }

  params.mSunAngularRadius              = settings.mSunAngularRadius;
  params.mBottomRadius                  = planetRadius;
  params.mTopRadius                     = atmosphereRadius;
  params.mGroundAlbedo                  = settings.mGroundAlbedo.get();
  params.mMaxSunZenithAngle             = 120.0 / 180.0 * glm::pi<double>();
  params.mSampleCountOpticalDepth       = settings.mSampleCountOpticalDepth.get();
  params.mSampleCountSingleScattering   = settings.mSampleCountSingleScattering.get();
  params.mSampleCountMultiScattering    = settings.mSampleCountMultiScattering.get();
  params.mSampleCountScatteringDensity  = settings.mSampleCountScatteringDensity.get();
  params.mSampleCountIndirectIrradiance = settings.mSampleCountIndirectIrradiance.get();
  params.mTransmittanceTextureWidth     = settings.mTransmittanceTextureWidth.get();
  params.mTransmittanceTextureHeight    = settings.mTransmittanceTextureHeight.get();
  params.mScatteringTextureRSize        = settings.mScatteringTextureRSize.get();
  params.mScatteringTextureMuSize       = settings.mScatteringTextureMuSize.get();
  params.mScatteringTextureMuSSize      = settings.mScatteringTextureMuSSize.get();
  params.mScatteringTextureNuSize       = settings.mScatteringTextureNuSize.get();
  params.mIrradianceTextureWidth        = settings.mIrradianceTextureWidth.get();
  params.mIrradianceTextureHeight       = settings.mIrradianceTextureHeight.get();

  mImpl.reset(new internal::Implementation(params));

  glDisable(GL_CULL_FACE);
  mImpl->Init(settings.mMultiScatteringOrder.get());
  glEnable(GL_CULL_FACE);

  return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

GLuint Model::getShader() const {
  return mImpl->shader();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

GLuint Model::setUniforms(GLuint program, GLuint startTextureUnit) const {
  mImpl->SetProgramUniforms(program, startTextureUnit, startTextureUnit + 1, startTextureUnit + 2,
      startTextureUnit + 3, startTextureUnit + 4);
  return startTextureUnit + 6;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::atmospheres::models::bruneton
