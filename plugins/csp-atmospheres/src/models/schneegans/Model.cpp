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

// The parameterization and comments below are based on the demo application by Eric Bruneton. The
// original source code can be found here:
// https://github.com/ebruneton/precomputed_atmospheric_scattering/blob/master/atmosphere/demo/demo.cc

enum class Luminance {
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

constexpr bool HALF_PRECISION = false;

// Values from "Reference Solar Spectral Irradiance: ASTM G-173", ETR column
// (see http://rredc.nrel.gov/solar/spectra/am1.5/ASTMG173/ASTMG173.html),
// summed and averaged in each bin (e.g. the value for 360nm is the average
// of the ASTM G-173 values for all wavelengths between 360 and 370nm).
// Values in W.m^-2.
constexpr int                              LAMBDA_MIN       = 360;
constexpr int                              LAMBDA_MAX       = 830;
constexpr int                              LAMBDA_STEPS     = 48;
constexpr std::array<double, LAMBDA_STEPS> SOLAR_IRRADIANCE = {1.11776, 1.14259, 1.01249, 1.14716,
    1.72765, 1.73054, 1.6887, 1.61253, 1.91198, 2.03474, 2.02042, 2.02212, 1.93377, 1.95809,
    1.91686, 1.8298, 1.8685, 1.8931, 1.85149, 1.8504, 1.8341, 1.8345, 1.8147, 1.78158, 1.7533,
    1.6965, 1.68194, 1.64654, 1.6048, 1.52143, 1.55622, 1.5113, 1.474, 1.4482, 1.41018, 1.36775,
    1.34188, 1.31429, 1.28303, 1.26758, 1.2367, 1.2082, 1.18737, 1.14683, 1.12362, 1.1058, 1.07124,
    1.04992};

////////////////////////////////////////////////////////////////////////////////////////////////////

void from_json(nlohmann::json const& j, Model::Settings::Layer& o) {
  cs::core::Settings::deserialize(j, "width", o.mWidth);
  cs::core::Settings::deserialize(j, "expTerm", o.mExpTerm);
  cs::core::Settings::deserialize(j, "expScale", o.mExpScale);
  cs::core::Settings::deserialize(j, "linearTerm", o.mLinearTerm);
  cs::core::Settings::deserialize(j, "constantTerm", o.mConstantTerm);
}

void to_json(nlohmann::json& j, Model::Settings::Layer const& o) {
  cs::core::Settings::serialize(j, "width", o.mWidth);
  cs::core::Settings::serialize(j, "expTerm", o.mExpTerm);
  cs::core::Settings::serialize(j, "expScale", o.mExpScale);
  cs::core::Settings::serialize(j, "linearTerm", o.mLinearTerm);
  cs::core::Settings::serialize(j, "constantTerm", o.mConstantTerm);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void from_json(nlohmann::json const& j, Model::Settings::Component& o) {
  cs::core::Settings::deserialize(j, "extinction", o.mExtinction);
  cs::core::Settings::deserialize(j, "phase", o.mPhase);
  cs::core::Settings::deserialize(j, "layers", o.mLayers);
}

void to_json(nlohmann::json& j, Model::Settings::Component const& o) {
  cs::core::Settings::serialize(j, "extinction", o.mExtinction);
  cs::core::Settings::serialize(j, "phase", o.mPhase);
  cs::core::Settings::serialize(j, "layers", o.mLayers);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void from_json(nlohmann::json const& j, Model::Settings& o) {
  cs::core::Settings::deserialize(j, "sunAngularRadius", o.mSunAngularRadius);
  cs::core::Settings::deserialize(j, "components", o.mComponents);
  cs::core::Settings::deserialize(j, "groundAlbedo", o.mGroundAlbedo);
}

void to_json(nlohmann::json& j, Model::Settings const& o) {
  cs::core::Settings::serialize(j, "sunAngularRadius", o.mSunAngularRadius);
  cs::core::Settings::serialize(j, "components", o.mComponents);
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

  auto createComponent = [](Settings::Component const& settings) {
    internal::AtmosphereComponent result;

    for (auto const& layer : settings.mLayers) {
      result.layers.emplace_back(
          layer.mWidth, layer.mExpTerm, layer.mExpScale, layer.mLinearTerm, layer.mConstantTerm);
    }

    internal::CSVLoader::readPhase(settings.mPhase, result);
    internal::CSVLoader::readExtinction(settings.mExtinction, result);

    return result;
  };

  auto rayleigh = createComponent(settings.mComponents[0]);
  auto mie      = createComponent(settings.mComponents[1]);
  auto ozone    = createComponent(settings.mComponents[2]);

  std::vector<double> wavelengths;
  std::vector<double> solarIrradiance;
  std::vector<double> groundAlbedo;

  for (int l = LAMBDA_MIN; l <= LAMBDA_MAX; l += 10) {
    wavelengths.push_back(l);
    solarIrradiance.push_back(SOLAR_IRRADIANCE[(l - LAMBDA_MIN) / 10]);
    groundAlbedo.push_back(settings.mGroundAlbedo.get());
  }

  Luminance luminanceMode     = Luminance::PRECOMPUTED;
  double    maxSunZenithAngle = (HALF_PRECISION ? 102.0 : 120.0) / 180.0 * glm::pi<double>();

  mModel.reset(new internal::Model(wavelengths, solarIrradiance, settings.mSunAngularRadius,
      planetRadius, atmosphereRadius, rayleigh, mie, ozone, groundAlbedo, maxSunZenithAngle, 1.0,
      luminanceMode == Luminance::PRECOMPUTED ? 15 : 3, HALF_PRECISION));

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
  mModel->SetProgramUniforms(
      program, startTextureUnit, startTextureUnit + 1, startTextureUnit + 2, startTextureUnit + 3);
  return startTextureUnit + 4;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::atmospheres::models::schneegans
