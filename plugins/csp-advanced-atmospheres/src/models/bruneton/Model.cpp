////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "Model.hpp"

#include <glm/gtc/constants.hpp>

namespace csp::atmospheres::models::bruneton {

////////////////////////////////////////////////////////////////////////////////////////////////////

enum Luminance {
  // Render the spectral radiance at kLambdaR, kLambdaG, kLambdaB.
  NONE,
  // Render the sRGB luminance, using an approximate (on the fly) conversion
  // from 3 spectral radiance values only (see section 14.3 in <a href=
  // "https://arxiv.org/pdf/1612.04336.pdf">A Qualitative and Quantitative
  //  Evaluation of 8 Clear Sky Models</a>).
  APPROXIMATE,
  // Render the sRGB luminance, precomputed from 15 spectral radiance values
  // (see section 4.4 in <a href=
  // "http://www.oskee.wz.cz/stranka/uploads/SCCG10ElekKmoch.pdf">Real-time
  //  Spectral Scattering in Large-scale Natural Participating Media</a>).
  PRECOMPUTED
};

////////////////////////////////////////////////////////////////////////////////////////////////////

void from_json(nlohmann::json const& j, Model::Settings& o) {
  //   cs::core::Settings::deserialize(j, "height", o.mHeight);
}

void to_json(nlohmann::json& j, Model::Settings const& o) {
  //   cs::core::Settings::serialize(j, "height", o.mHeight);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool Model::init(nlohmann::json modelSettings, double planetRadius) {

  if (mPreviousSettings == modelSettings && mPlanetRadius == planetRadius) {
    return false;
  }

  mPreviousSettings = std::move(modelSettings);
  mSettings         = mPreviousSettings;
  mPlanetRadius     = planetRadius;

  constexpr double kSunAngularRadius = 0.00935 / 2.0;

  constexpr bool      use_half_precision_    = false;
  constexpr bool      use_ozone_             = true;
  constexpr bool      use_combined_textures_ = true;
  constexpr Luminance use_luminance_         = Luminance::PRECOMPUTED;

  // Values from "Reference Solar Spectral Irradiance: ASTM G-173", ETR column
  // (see http://rredc.nrel.gov/solar/spectra/am1.5/ASTMG173/ASTMG173.html),
  // summed and averaged in each bin (e.g. the value for 360nm is the average
  // of the ASTM G-173 values for all wavelengths between 360 and 370nm).
  // Values in W.m^-2.
  constexpr int    kLambdaMin           = 360;
  constexpr int    kLambdaMax           = 830;
  constexpr double kSolarIrradiance[48] = {1.11776, 1.14259, 1.01249, 1.14716, 1.72765, 1.73054,
      1.6887, 1.61253, 1.91198, 2.03474, 2.02042, 2.02212, 1.93377, 1.95809, 1.91686, 1.8298,
      1.8685, 1.8931, 1.85149, 1.8504, 1.8341, 1.8345, 1.8147, 1.78158, 1.7533, 1.6965, 1.68194,
      1.64654, 1.6048, 1.52143, 1.55622, 1.5113, 1.474, 1.4482, 1.41018, 1.36775, 1.34188, 1.31429,
      1.28303, 1.26758, 1.2367, 1.2082, 1.18737, 1.14683, 1.12362, 1.1058, 1.07124, 1.04992};
  // Values from http://www.iup.uni-bremen.de/gruppen/molspec/databases/
  // referencespectra/o3spectra2011/index.html for 233K, summed and averaged in
  // each bin (e.g. the value for 360nm is the average of the original values
  // for all wavelengths between 360 and 370nm). Values in m^2.
  constexpr double kOzoneCrossSection[48] = {1.18e-27, 2.182e-28, 2.818e-28, 6.636e-28, 1.527e-27,
      2.763e-27, 5.52e-27, 8.451e-27, 1.582e-26, 2.316e-26, 3.669e-26, 4.924e-26, 7.752e-26,
      9.016e-26, 1.48e-25, 1.602e-25, 2.139e-25, 2.755e-25, 3.091e-25, 3.5e-25, 4.266e-25,
      4.672e-25, 4.398e-25, 4.701e-25, 5.019e-25, 4.305e-25, 3.74e-25, 3.215e-25, 2.662e-25,
      2.238e-25, 1.852e-25, 1.473e-25, 1.209e-25, 9.423e-26, 7.455e-26, 6.566e-26, 5.105e-26,
      4.15e-26, 4.228e-26, 3.237e-26, 2.451e-26, 2.801e-26, 2.534e-26, 1.624e-26, 1.465e-26,
      2.078e-26, 1.383e-26, 7.105e-27};
  // From https://en.wikipedia.org/wiki/Dobson_unit, in molecules.m^-2.
  constexpr double kDobsonUnit = 2.687e20;
  // Maximum number density of ozone molecules, in m^-3 (computed so at to get
  // 300 Dobson units of ozone - for this we divide 300 DU by the integral of
  // the ozone density profile defined below, which is equal to 15km).
  constexpr double kMaxOzoneNumberDensity     = 300.0 * kDobsonUnit / 15000.0;
  constexpr double kBottomRadius              = 6371000.0;
  constexpr double kTopRadius                 = 6520000.0;
  constexpr double kRayleigh                  = 1.24062e-6;
  constexpr double kRayleighScaleHeight       = 8000.0;
  constexpr double kMieScaleHeight            = 1200.0;
  constexpr double kMieAngstromAlpha          = 0.0;
  constexpr double kMieAngstromBeta           = 5.328e-3;
  constexpr double kMieSingleScatteringAlbedo = 0.9;
  constexpr double kMiePhaseFunctionG         = 0.8;
  constexpr double kGroundAlbedo              = 0.1;
  const double     max_sun_zenith_angle =
      (use_half_precision_ ? 102.0 : 120.0) / 180.0 * glm::pi<double>();

  internal::DensityProfileLayer rayleigh_layer(0.0, 1.0, -1.0 / kRayleighScaleHeight, 0.0, 0.0);
  internal::DensityProfileLayer mie_layer(0.0, 1.0, -1.0 / kMieScaleHeight, 0.0, 0.0);
  // Density profile increasing linearly from 0 to 1 between 10 and 25km, and
  // decreasing linearly from 1 to 0 between 25 and 40km. This is an approximate
  // profile from http://www.kln.ac.lk/science/Chemistry/Teaching_Resources/
  // Documents/Introduction%20to%20atmospheric%20chemistry.pdf (page 10).
  std::vector<internal::DensityProfileLayer> ozone_density;
  ozone_density.push_back(
      internal::DensityProfileLayer(25000.0, 0.0, 0.0, 1.0 / 15000.0, -2.0 / 3.0));
  ozone_density.push_back(internal::DensityProfileLayer(0.0, 0.0, 0.0, -1.0 / 15000.0, 8.0 / 3.0));

  std::vector<double> wavelengths;
  std::vector<double> solar_irradiance;
  std::vector<double> rayleigh_scattering;
  std::vector<double> mie_scattering;
  std::vector<double> mie_extinction;
  std::vector<double> absorption_extinction;
  std::vector<double> ground_albedo;
  for (int l = kLambdaMin; l <= kLambdaMax; l += 10) {
    double lambda = static_cast<double>(l) * 1e-3; // micro-meters
    double mie    = kMieAngstromBeta / kMieScaleHeight * pow(lambda, -kMieAngstromAlpha);
    wavelengths.push_back(l);
    solar_irradiance.push_back(kSolarIrradiance[(l - kLambdaMin) / 10]);
    rayleigh_scattering.push_back(kRayleigh * pow(lambda, -4));
    mie_scattering.push_back(mie * kMieSingleScatteringAlbedo);
    mie_extinction.push_back(mie);
    absorption_extinction.push_back(
        use_ozone_ ? kMaxOzoneNumberDensity * kOzoneCrossSection[(l - kLambdaMin) / 10] : 0.0);
    ground_albedo.push_back(kGroundAlbedo);
  }

  mModel.reset(new internal::Model(wavelengths, solar_irradiance, kSunAngularRadius, kBottomRadius,
      kTopRadius, {rayleigh_layer}, rayleigh_scattering, {mie_layer}, mie_scattering,
      mie_extinction, kMiePhaseFunctionG, ozone_density, absorption_extinction, ground_albedo,
      max_sun_zenith_angle, 1.0, use_luminance_ == PRECOMPUTED ? 15 : 3, use_combined_textures_,
      use_half_precision_));
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
  mModel->SetProgramUniforms(program, startTextureUnit, startTextureUnit + 1, startTextureUnit + 2);
  return startTextureUnit + 3;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::atmospheres::models::bruneton
