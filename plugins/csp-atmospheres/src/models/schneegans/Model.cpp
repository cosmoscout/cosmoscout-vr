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

namespace csp::atmospheres::models::schneegans {

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
  cs::core::Settings::deserialize(j, "particles_a", o.mParticlesA);
  cs::core::Settings::deserialize(j, "particles_b", o.mParticlesB);
  cs::core::Settings::deserialize(j, "absorbing_particles", o.mAbsorbingParticles);
  cs::core::Settings::deserialize(j, "groundAlbedo", o.mGroundAlbedo);
}

void to_json(nlohmann::json& j, Model::Settings const& o) {
  cs::core::Settings::serialize(j, "sunAngularRadius", o.mSunAngularRadius);
  cs::core::Settings::serialize(j, "particles_a", o.mParticlesA);
  cs::core::Settings::serialize(j, "particles_b", o.mParticlesB);
  cs::core::Settings::serialize(j, "absorbing_particles", o.mAbsorbingParticles);
  cs::core::Settings::serialize(j, "groundAlbedo", o.mGroundAlbedo);
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

  internal::ScatteringAtmosphereComponent rayleigh;
  internal::ScatteringAtmosphereComponent mie;
  internal::AbsorbingAtmosphereComponent  ozone;

  std::vector<double> wavelengths;
  uint32_t            densityCount = 0;

  rayleigh.density = internal::CSVLoader::readDensity(settings.mParticlesA.mDensity, densityCount);
  rayleigh.phase   = internal::CSVLoader::readPhase(settings.mParticlesA.mPhase, wavelengths);
  rayleigh.scattering =
      internal::CSVLoader::readExtinction(settings.mParticlesA.mBetaSca, wavelengths);
  rayleigh.absorption =
      internal::CSVLoader::readExtinction(settings.mParticlesA.mBetaAbs, wavelengths);

  mie.density    = internal::CSVLoader::readDensity(settings.mParticlesB.mDensity, densityCount);
  mie.phase      = internal::CSVLoader::readPhase(settings.mParticlesB.mPhase, wavelengths);
  mie.scattering = internal::CSVLoader::readExtinction(settings.mParticlesB.mBetaSca, wavelengths);
  mie.absorption = internal::CSVLoader::readExtinction(settings.mParticlesB.mBetaAbs, wavelengths);

  if (settings.mAbsorbingParticles) {

    ozone.density =
        internal::CSVLoader::readDensity(settings.mAbsorbingParticles->mDensity, densityCount);
    ozone.absorption =
        internal::CSVLoader::readExtinction(settings.mAbsorbingParticles->mBetaAbs, wavelengths);

  } else {
    ozone.density    = std::vector<double>(densityCount, 0.0);
    ozone.absorption = std::vector<double>(wavelengths.size(), 0.0);
  }

  if (wavelengths.size() < 3) {
    throw std::runtime_error(
        "At least three different wavelengths should be given in the scattering data!");
  } else if (wavelengths.size() == 3 && (wavelengths[0] != internal::Model::kLambdaB ||
                                            wavelengths[1] != internal::Model::kLambdaG ||
                                            wavelengths[2] != internal::Model::kLambdaR)) {
    throw std::runtime_error("If three different wavelengths are given in the scattering data, "
                             "they should be exactly for 440 nm, 550 nm, and 680 nm!");
  }

  double maxSunZenithAngle = 120.0 / 180.0 * glm::pi<double>();

  mModel.reset(
      new internal::Model(wavelengths, settings.mSunAngularRadius, planetRadius, atmosphereRadius,
          rayleigh, mie, ozone, settings.mGroundAlbedo.get(), maxSunZenithAngle, 1.0));

  glDisable(GL_CULL_FACE);
  mModel->Init();
  glEnable(GL_CULL_FACE);

  return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

GLuint Model::getShader() const {
  return mModel->shader();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

GLuint Model::setUniforms(GLuint program, GLuint startTextureUnit) const {
  mModel->SetProgramUniforms(program, startTextureUnit, startTextureUnit + 1, startTextureUnit + 2,
      startTextureUnit + 3, startTextureUnit + 4);
  return startTextureUnit + 6;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::atmospheres::models::schneegans
