////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-FileCopyrightText: 2017 Eric Bruneton
// SPDX-License-Identifier: BSD-3-Clause

#include "Model.hpp"

#include "../../../src/cs-utils/filesystem.hpp"
#include "../../../src/cs-utils/utils.hpp"
#include "../../logger.hpp"

#include <cassert>
#include <cmath>
#include <glm/gtc/type_ptr.hpp>
#include <iostream>
#include <memory>
#include <tiffio.h>

#include <glm/gtc/constants.hpp>

// This file is based in large parts on the original implementation by Eric Bruneton:
// https://github.com/ebruneton/precomputed_atmospheric_scattering/blob/master/atmosphere/model.cc

// While implementing the atmospheric model into CosmoScout VR, we have refactored some parts of the
// code, however this is mostly related to how variables are named and how input parameters are
// passed to the model. The only fundamental change is that the phase functions for aerosols and
// molecules as well as their density distributions are now loaded from CSV files and then later
// sampled from textures.

// Below, we will indicate for each group of function whether something has been changed and a link
// to the original explanations of the methods by Eric Bruneton.

namespace csp::atmospheres::models::bruneton {

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace {
// Values from "Reference Solar Spectral Irradiance: ASTM G-173", ETR column  (see
// http://rredc.nrel.gov/solar/spectra/am1.5/ASTMG173/ASTMG173.html), summed and averaged in each
// bin (e.g. the value for 360nm is the average of the ASTM G-173 values for all wavelengths between
// 360 and 370nm). Values in W.m^-2. Copied from:
// https://github.com/ebruneton/precomputed_atmospheric_scattering/blob/master/atmosphere/demo/demo.cc
// clang-format off
const std::vector<float> SOLAR_IRRADIANCE = {
                                                                1.11776F, 1.14259F, 1.01249F, 1.14716F,
    1.72765F, 1.73054F, 1.6887F,  1.61253F, 1.91198F, 2.03474F, 2.02042F, 2.02212F, 1.93377F, 1.95809F,
    1.91686F, 1.8298F,  1.8685F,  1.8931F,  1.85149F, 1.8504F,  1.8341F,  1.8345F,  1.8147F,  1.78158F,
    1.7533F,  1.6965F,  1.68194F, 1.64654F, 1.6048F,  1.52143F, 1.55622F, 1.5113F,  1.474F,   1.4482F,
    1.41018F, 1.36775F, 1.34188F, 1.31429F, 1.28303F, 1.26758F, 1.2367F,  1.2082F,  1.18737F, 1.14683F,
    1.12362F, 1.1058F,  1.07124F, 1.04992F
};

const std::vector<float> WAVELENGTHS = {
                                              360.F, 370.F, 380.F, 390.F,
    400.F, 410.F, 420.F, 430.F, 440.F, 450.F, 460.F, 470.F, 480.F, 490.F,
    500.F, 510.F, 520.F, 530.F, 540.F, 550.F, 560.F, 570.F, 580.F, 590.F,
    600.F, 610.F, 620.F, 630.F, 640.F, 650.F, 660.F, 670.F, 680.F, 690.F,
    700.F, 710.F, 720.F, 730.F, 740.F, 750.F, 760.F, 770.F, 780.F, 790.F,
    800.F, 810.F, 820.F, 830.F
};
// clang-format on

// The conversion factor between watts and lumens.
constexpr float MAX_LUMINOUS_EFFICACY = 683.0;

// Values from "CIE (1931) 2-deg color matching functions", see
// "http://web.archive.org/web/20081228084047/http://www.cvrl.org/database/data/cmfs/ciexyz31.txt".
// Copied from:
// https://github.com/ebruneton/precomputed_atmospheric_scattering/blob/master/atmosphere/constants.h
// clang-format off
constexpr float CIE_2_DEG_COLOR_MATCHING_FUNCTIONS[380] = {
    360.F, 0.000129900000F, 0.000003917000F, 0.000606100000F,
    365.F, 0.000232100000F, 0.000006965000F, 0.001086000000F,
    370.F, 0.000414900000F, 0.000012390000F, 0.001946000000F,
    375.F, 0.000741600000F, 0.000022020000F, 0.003486000000F,
    380.F, 0.001368000000F, 0.000039000000F, 0.006450001000F,
    385.F, 0.002236000000F, 0.000064000000F, 0.010549990000F,
    390.F, 0.004243000000F, 0.000120000000F, 0.020050010000F,
    395.F, 0.007650000000F, 0.000217000000F, 0.036210000000F,
    400.F, 0.014310000000F, 0.000396000000F, 0.067850010000F,
    405.F, 0.023190000000F, 0.000640000000F, 0.110200000000F,
    410.F, 0.043510000000F, 0.001210000000F, 0.207400000000F,
    415.F, 0.077630000000F, 0.002180000000F, 0.371300000000F,
    420.F, 0.134380000000F, 0.004000000000F, 0.645600000000F,
    425.F, 0.214770000000F, 0.007300000000F, 1.039050100000F,
    430.F, 0.283900000000F, 0.011600000000F, 1.385600000000F,
    435.F, 0.328500000000F, 0.016840000000F, 1.622960000000F,
    440.F, 0.348280000000F, 0.023000000000F, 1.747060000000F,
    445.F, 0.348060000000F, 0.029800000000F, 1.782600000000F,
    450.F, 0.336200000000F, 0.038000000000F, 1.772110000000F,
    455.F, 0.318700000000F, 0.048000000000F, 1.744100000000F,
    460.F, 0.290800000000F, 0.060000000000F, 1.669200000000F,
    465.F, 0.251100000000F, 0.073900000000F, 1.528100000000F,
    470.F, 0.195360000000F, 0.090980000000F, 1.287640000000F,
    475.F, 0.142100000000F, 0.112600000000F, 1.041900000000F,
    480.F, 0.095640000000F, 0.139020000000F, 0.812950100000F,
    485.F, 0.057950010000F, 0.169300000000F, 0.616200000000F,
    490.F, 0.032010000000F, 0.208020000000F, 0.465180000000F,
    495.F, 0.014700000000F, 0.258600000000F, 0.353300000000F,
    500.F, 0.004900000000F, 0.323000000000F, 0.272000000000F,
    505.F, 0.002400000000F, 0.407300000000F, 0.212300000000F,
    510.F, 0.009300000000F, 0.503000000000F, 0.158200000000F,
    515.F, 0.029100000000F, 0.608200000000F, 0.111700000000F,
    520.F, 0.063270000000F, 0.710000000000F, 0.078249990000F,
    525.F, 0.109600000000F, 0.793200000000F, 0.057250010000F,
    530.F, 0.165500000000F, 0.862000000000F, 0.042160000000F,
    535.F, 0.225749900000F, 0.914850100000F, 0.029840000000F,
    540.F, 0.290400000000F, 0.954000000000F, 0.020300000000F,
    545.F, 0.359700000000F, 0.980300000000F, 0.013400000000F,
    550.F, 0.433449900000F, 0.994950100000F, 0.008749999000F,
    555.F, 0.512050100000F, 1.000000000000F, 0.005749999000F,
    560.F, 0.594500000000F, 0.995000000000F, 0.003900000000F,
    565.F, 0.678400000000F, 0.978600000000F, 0.002749999000F,
    570.F, 0.762100000000F, 0.952000000000F, 0.002100000000F,
    575.F, 0.842500000000F, 0.915400000000F, 0.001800000000F,
    580.F, 0.916300000000F, 0.870000000000F, 0.001650001000F,
    585.F, 0.978600000000F, 0.816300000000F, 0.001400000000F,
    590.F, 1.026300000000F, 0.757000000000F, 0.001100000000F,
    595.F, 1.056700000000F, 0.694900000000F, 0.001000000000F,
    600.F, 1.062200000000F, 0.631000000000F, 0.000800000000F,
    605.F, 1.045600000000F, 0.566800000000F, 0.000600000000F,
    610.F, 1.002600000000F, 0.503000000000F, 0.000340000000F,
    615.F, 0.938400000000F, 0.441200000000F, 0.000240000000F,
    620.F, 0.854449900000F, 0.381000000000F, 0.000190000000F,
    625.F, 0.751400000000F, 0.321000000000F, 0.000100000000F,
    630.F, 0.642400000000F, 0.265000000000F, 0.000049999990F,
    635.F, 0.541900000000F, 0.217000000000F, 0.000030000000F,
    640.F, 0.447900000000F, 0.175000000000F, 0.000020000000F,
    645.F, 0.360800000000F, 0.138200000000F, 0.000010000000F,
    650.F, 0.283500000000F, 0.107000000000F, 0.000000000000F,
    655.F, 0.218700000000F, 0.081600000000F, 0.000000000000F,
    660.F, 0.164900000000F, 0.061000000000F, 0.000000000000F,
    665.F, 0.121200000000F, 0.044580000000F, 0.000000000000F,
    670.F, 0.087400000000F, 0.032000000000F, 0.000000000000F,
    675.F, 0.063600000000F, 0.023200000000F, 0.000000000000F,
    680.F, 0.046770000000F, 0.017000000000F, 0.000000000000F,
    685.F, 0.032900000000F, 0.011920000000F, 0.000000000000F,
    690.F, 0.022700000000F, 0.008210000000F, 0.000000000000F,
    695.F, 0.015840000000F, 0.005723000000F, 0.000000000000F,
    700.F, 0.011359160000F, 0.004102000000F, 0.000000000000F,
    705.F, 0.008110916000F, 0.002929000000F, 0.000000000000F,
    710.F, 0.005790346000F, 0.002091000000F, 0.000000000000F,
    715.F, 0.004109457000F, 0.001484000000F, 0.000000000000F,
    720.F, 0.002899327000F, 0.001047000000F, 0.000000000000F,
    725.F, 0.002049190000F, 0.000740000000F, 0.000000000000F,
    730.F, 0.001439971000F, 0.000520000000F, 0.000000000000F,
    735.F, 0.000999949300F, 0.000361100000F, 0.000000000000F,
    740.F, 0.000690078600F, 0.000249200000F, 0.000000000000F,
    745.F, 0.000476021300F, 0.000171900000F, 0.000000000000F,
    750.F, 0.000332301100F, 0.000120000000F, 0.000000000000F,
    755.F, 0.000234826100F, 0.000084800000F, 0.000000000000F,
    760.F, 0.000166150500F, 0.000060000000F, 0.000000000000F,
    765.F, 0.000117413000F, 0.000042400000F, 0.000000000000F,
    770.F, 0.000083075270F, 0.000030000000F, 0.000000000000F,
    775.F, 0.000058706520F, 0.000021200000F, 0.000000000000F,
    780.F, 0.000041509940F, 0.000014990000F, 0.000000000000F,
    785.F, 0.000029353260F, 0.000010600000F, 0.000000000000F,
    790.F, 0.000020673830F, 0.000007465700F, 0.000000000000F,
    795.F, 0.000014559770F, 0.000005257800F, 0.000000000000F,
    800.F, 0.000010253980F, 0.000003702900F, 0.000000000000F,
    805.F, 0.000007221456F, 0.000002607800F, 0.000000000000F,
    810.F, 0.000005085868F, 0.000001836600F, 0.000000000000F,
    815.F, 0.000003581652F, 0.000001293400F, 0.000000000000F,
    820.F, 0.000002522525F, 0.000000910930F, 0.000000000000F,
    825.F, 0.000001776509F, 0.000000641530F, 0.000000000000F,
    830.F, 0.000001251141F, 0.000000451810F, 0.000000000000F,
};
// clang-format on

// The conversion matrix from XYZ to linear sRGB color spaces.
// Values from https://en.wikipedia.org/wiki/SRGB.
// clang-format off
constexpr float XYZ_TO_SRGB[9] = {
    +3.2406F, -1.5372F, -0.4986F,
    -0.9689F, +1.8758F, +0.0415F,
    +0.0557F, -0.2040F, +1.0570F
};
// clang-format on

// Utility classes and Functions -------------------------------------------------------------------

////////////////////////////////////////////////////////////////////////////////////////////////////

// These GL-texture generators have been made a bit more flexible to allow for different pixel
// formats. At the same time, we have removed support for the half-resolution pixel formats (they
// quickly lead to artefacts with the high-dynamic range of CosmoScout VR). Also, we do not check
// for the availability of RGB textures anymore but use RGB textures everywhere.

GLuint NewTexture2d(int width, int height, GLenum internalFormat, GLenum format, GLenum type,
    void* data = nullptr) {
  GLuint texture;
  glGenTextures(1, &texture);
  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, texture);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
  glTexImage2D(GL_TEXTURE_2D, 0, internalFormat, width, height, 0, format, type, data);
  return texture;
}

GLuint NewTexture3d(int width, int height, int depth, GLenum internalFormat, GLenum format,
    GLenum type, void* data = nullptr) {
  GLuint texture;
  glGenTextures(1, &texture);
  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_3D, texture);
  glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
  glTexImage3D(GL_TEXTURE_3D, 0, internalFormat, width, height, depth, 0, format, type, data);
  return texture;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// This is functionality-wise identical to the original implementation.

float CieColorMatchingFunctionTableValue(float wavelength, int column) {
  if (wavelength <= WAVELENGTHS.front() || wavelength >= WAVELENGTHS.back()) {
    return 0.F;
  }
  float u   = (wavelength - WAVELENGTHS.front()) / 5.F;
  int   row = static_cast<int>(std::floor(u));
  assert(row >= 0 && row + 1 < 95);
  assert(CIE_2_DEG_COLOR_MATCHING_FUNCTIONS[4 * row] <= wavelength &&
         CIE_2_DEG_COLOR_MATCHING_FUNCTIONS[4 * (row + 1)] >= wavelength);
  u -= row;
  return CIE_2_DEG_COLOR_MATCHING_FUNCTIONS[4 * row + column] * (1.F - u) +
         CIE_2_DEG_COLOR_MATCHING_FUNCTIONS[4 * (row + 1) + column] * u;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// This is functionality-wise identical to the original implementation.

float Interpolate(std::vector<float> const& xVals, std::vector<float> const& yVals, float x) {
  assert(yVals.size() == xVals.size());

  if (x < xVals[0]) {
    return yVals[0];
  }

  for (unsigned int i = 0; i < xVals.size() - 1; ++i) {
    if (x < xVals[i + 1]) {
      float u = (x - xVals[i]) / (xVals[i + 1] - xVals[i]);
      return yVals[i] * (1.F - u) + yVals[i + 1] * u;
    }
  }

  return yVals[yVals.size() - 1];
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// This is functionality-wise identical to the original implementation.

void ComputeSpectralRadianceToLuminanceFactors(float lambdaPower, float* kR, float* kG, float* kB) {

  *kR           = 0.F;
  *kG           = 0.F;
  *kB           = 0.F;
  float solarR  = Interpolate(WAVELENGTHS, SOLAR_IRRADIANCE, Model::kLambdaR);
  float solarG  = Interpolate(WAVELENGTHS, SOLAR_IRRADIANCE, Model::kLambdaG);
  float solarB  = Interpolate(WAVELENGTHS, SOLAR_IRRADIANCE, Model::kLambdaB);
  float dLambda = 1.0;
  for (float lambda = WAVELENGTHS.front(); lambda <= WAVELENGTHS.back(); lambda += dLambda) {
    float        x_bar      = CieColorMatchingFunctionTableValue(lambda, 1);
    float        y_bar      = CieColorMatchingFunctionTableValue(lambda, 2);
    float        z_bar      = CieColorMatchingFunctionTableValue(lambda, 3);
    const float* xyz2srgb   = XYZ_TO_SRGB;
    float        r_bar      = xyz2srgb[0] * x_bar + xyz2srgb[1] * y_bar + xyz2srgb[2] * z_bar;
    float        g_bar      = xyz2srgb[3] * x_bar + xyz2srgb[4] * y_bar + xyz2srgb[5] * z_bar;
    float        b_bar      = xyz2srgb[6] * x_bar + xyz2srgb[7] * y_bar + xyz2srgb[8] * z_bar;
    float        irradiance = Interpolate(WAVELENGTHS, SOLAR_IRRADIANCE, lambda);
    *kR += r_bar * irradiance / solarR * pow(lambda / Model::kLambdaR, lambdaPower);
    *kG += g_bar * irradiance / solarG * pow(lambda / Model::kLambdaG, lambdaPower);
    *kB += b_bar * irradiance / solarB * pow(lambda / Model::kLambdaB, lambdaPower);
  }
  *kR *= MAX_LUMINOUS_EFFICACY * dLambda;
  *kG *= MAX_LUMINOUS_EFFICACY * dLambda;
  *kB *= MAX_LUMINOUS_EFFICACY * dLambda;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// The functions below are used to inject the atmosphere components into the shader source code.
// These were not present in the original implementation and have been added because we refactored
// how data is passed to the shader.

// This first method is used to create a GLSL "vec3(...)" string based on a linearly interpolated 1D
// function. The function is defined by the first two parameters and the three values are extracted
// by linear interpolation using the three values passed in as last parameter.
std::string extractVec3(
    std::vector<float> const& xVals, std::vector<float> const& yVals, glm::vec3 const& lambdas) {
  float r = Interpolate(xVals, yVals, lambdas[0]);
  float g = Interpolate(xVals, yVals, lambdas[1]);
  float b = Interpolate(xVals, yVals, lambdas[2]);
  return "vec3(" + cs::utils::toString(r) + "," + cs::utils::toString(g) + "," +
         cs::utils::toString(b) + ")";
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace

////////////////////////////////////////////////////////////////////////////////////////////////////

void from_json(nlohmann::json const& j, Model::Settings& o) {
  cs::core::Settings::deserialize(j, "sunAngularRadius", o.mSunAngularRadius);
  cs::core::Settings::deserialize(j, "phaseTexture", o.mPhaseTexture);
  cs::core::Settings::deserialize(j, "transmittanceTexture", o.mTransmittanceTexture);
  cs::core::Settings::deserialize(j, "irradianceTexture", o.mIrradianceTexture);
  cs::core::Settings::deserialize(j, "singleScatteringTexture", o.mSingleScatteringTexture);
  cs::core::Settings::deserialize(j, "multipleScatteringTexture", o.mMultipleScatteringTexture);
  cs::core::Settings::deserialize(j, "transmittanceTextureWidth", o.mTransmittanceTextureWidth);
  cs::core::Settings::deserialize(j, "transmittanceTextureHeight", o.mTransmittanceTextureHeight);
  cs::core::Settings::deserialize(j, "scatteringTextureRSize", o.mScatteringTextureRSize);
  cs::core::Settings::deserialize(j, "scatteringTextureMuSize", o.mScatteringTextureMuSize);
  cs::core::Settings::deserialize(j, "scatteringTextureMuSSize", o.mScatteringTextureMuSSize);
  cs::core::Settings::deserialize(j, "scatteringTextureNuSize", o.mScatteringTextureNuSize);
  cs::core::Settings::deserialize(j, "irradianceTextureWidth", o.mIrradianceTextureWidth);
  cs::core::Settings::deserialize(j, "irradianceTextureHeight", o.mIrradianceTextureHeight);
  cs::core::Settings::deserialize(j, "maxSunZenithAngle", o.mMaxSunZenithAngle);
}

void to_json(nlohmann::json& j, Model::Settings const& o) {
  cs::core::Settings::serialize(j, "sunAngularRadius", o.mSunAngularRadius);
  cs::core::Settings::serialize(j, "phaseTexture", o.mPhaseTexture);
  cs::core::Settings::serialize(j, "transmittanceTexture", o.mTransmittanceTexture);
  cs::core::Settings::serialize(j, "irradianceTexture", o.mIrradianceTexture);
  cs::core::Settings::serialize(j, "singleScatteringTexture", o.mSingleScatteringTexture);
  cs::core::Settings::serialize(j, "multipleScatteringTexture", o.mMultipleScatteringTexture);
  cs::core::Settings::serialize(j, "transmittanceTextureWidth", o.mTransmittanceTextureWidth);
  cs::core::Settings::serialize(j, "transmittanceTextureHeight", o.mTransmittanceTextureHeight);
  cs::core::Settings::serialize(j, "scatteringTextureRSize", o.mScatteringTextureRSize);
  cs::core::Settings::serialize(j, "scatteringTextureMuSize", o.mScatteringTextureMuSize);
  cs::core::Settings::serialize(j, "scatteringTextureMuSSize", o.mScatteringTextureMuSSize);
  cs::core::Settings::serialize(j, "scatteringTextureNuSize", o.mScatteringTextureNuSize);
  cs::core::Settings::serialize(j, "irradianceTextureWidth", o.mIrradianceTextureWidth);
  cs::core::Settings::serialize(j, "irradianceTextureHeight", o.mIrradianceTextureHeight);
  cs::core::Settings::serialize(j, "maxSunZenithAngle", o.mMaxSunZenithAngle);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

Model::~Model() {
  glDeleteTextures(1, &mPhaseTexture);
  glDeleteTextures(1, &mTransmittanceTexture);
  glDeleteTextures(1, &mMultipleScatteringTexture);
  glDeleteTextures(1, &mSingleAerosolsScatteringTexture);
  glDeleteTextures(1, &mIrradianceTexture);
  glDeleteShader(mAtmosphereShader);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// The code below roughly follows the original implementation by Eric Bruneton.

// The original explanation of the methods still applies in most parts and is available online:
// https://ebruneton.github.io/precomputed_atmospheric_scattering/atmosphere/model.cc.html#implementation

// The main differences in the constructor are that we pass significantly more constants to the
// shader code as all the sampling counts are configurable now. Also, we create the new density
// texture which contains the density profiles for all atmosphere constituents.
bool Model::init(
    nlohmann::json const& modelSettings, double planetRadius, double atmosphereRadius) {

  Settings settings;

  try {
    settings = modelSettings;
  } catch (std::exception const& e) {
    logger().error("Failed to parse atmosphere parameters: {}", e.what());
  }

  // Compute the values for the SKY_RADIANCE_TO_LUMINANCE constant. In theory this should be 1 in
  // precomputed illuminance mode (because the precomputed textures already contain illuminance
  // values). In practice, however, storing true illuminance values in half precision textures
  // yields artefacts (because the values are too large), so we store illuminance values divided by
  // MAX_LUMINOUS_EFFICACY instead. This is why, in precomputed illuminance mode, we set
  // SKY_RADIANCE_TO_LUMINANCE to MAX_LUMINOUS_EFFICACY.
  bool  precomputeIlluminance = true; // settings.mWavelengths.size() > 3;
  float skyKR, skyKG, skyKB;
  if (precomputeIlluminance) {
    skyKR = skyKG = skyKB = MAX_LUMINOUS_EFFICACY;
  } else {
    ComputeSpectralRadianceToLuminanceFactors(-3 /* lambdaPower */, &skyKR, &skyKG, &skyKB);
  }
  // Compute the values for the SUN_RADIANCE_TO_LUMINANCE constant.
  float sunKR, sunKG, sunKB;
  ComputeSpectralRadianceToLuminanceFactors(0 /* lambdaPower */, &sunKR, &sunKG, &sunKB);

  // A lambda that creates a GLSL header containing our atmosphere computation functions,
  // specialized for the given atmosphere parameters and for the 3 wavelengths in 'lambdas'.
  auto model = cs::utils::filesystem::loadToString(
      "../share/resources/shaders/csp-atmospheres/models/bruneton/model.glsl");

  // clang-format off
  std::string shader =
    std::string("#version 330\n") +
    "const int TRANSMITTANCE_TEXTURE_WIDTH = "      + cs::utils::toString(settings.mTransmittanceTextureWidth) + ";\n" +
    "const int TRANSMITTANCE_TEXTURE_HEIGHT = "     + cs::utils::toString(settings.mTransmittanceTextureHeight) + ";\n" +
    "const int SCATTERING_TEXTURE_R_SIZE = "        + cs::utils::toString(settings.mScatteringTextureRSize) + ";\n" +
    "const int SCATTERING_TEXTURE_MU_SIZE = "       + cs::utils::toString(settings.mScatteringTextureMuSize) + ";\n" +
    "const int SCATTERING_TEXTURE_MU_S_SIZE = "     + cs::utils::toString(settings.mScatteringTextureMuSSize) + ";\n" +
    "const int SCATTERING_TEXTURE_NU_SIZE = "       + cs::utils::toString(settings.mScatteringTextureNuSize) + ";\n" +
    "const int IRRADIANCE_TEXTURE_WIDTH = "         + cs::utils::toString(settings.mIrradianceTextureWidth) + ";\n" +
    "const int IRRADIANCE_TEXTURE_HEIGHT = "        + cs::utils::toString(settings.mIrradianceTextureHeight) + ";\n" +
    "const vec3 SKY_SPECTRAL_RADIANCE_TO_LUMINANCE = vec3(" + cs::utils::toString(skyKR) + "," + cs::utils::toString(skyKG) + "," + cs::utils::toString(skyKB) + ");\n" +
    "const vec3 SUN_SPECTRAL_RADIANCE_TO_LUMINANCE = vec3(" + cs::utils::toString(sunKR) + "," + cs::utils::toString(sunKG) + "," + cs::utils::toString(sunKB) + ");\n" +
    "const vec3 SOLAR_IRRADIANCE = "                + extractVec3(WAVELENGTHS, SOLAR_IRRADIANCE, {kLambdaR, kLambdaG, kLambdaB}) + ";\n" +
    "const float SUN_ANGULAR_RADIUS = "             + cs::utils::toString(settings.mSunAngularRadius) + ";\n" +
    "const float BOTTOM_RADIUS = "                  + cs::utils::toString(planetRadius) + ";\n" +
    "const float TOP_RADIUS = "                     + cs::utils::toString(atmosphereRadius) + ";\n" +
    "const float MU_S_MIN = "                       + cs::utils::toString(std::cos(settings.mMaxSunZenithAngle.get()))+ ";\n" +
    "const float MOLECULES_PHASE_FUNCTION_V = 0.0;\n" +
    "const float AEROSOLS_PHASE_FUNCTION_V = 1.0;\n" +
    model;
  // clang-format on

  auto read2D = [](std::string const& path) {
    auto* data = TIFFOpen(path.c_str(), "r");

    if (!data) {
      logger().error("Failed to open TIFF file '{}'", path);
      return 0u;
    }

    uint32_t width{};
    uint32_t height{};

    TIFFGetField(data, TIFFTAG_IMAGELENGTH, &height);
    TIFFGetField(data, TIFFTAG_IMAGEWIDTH, &width);

    std::vector<float> pixels(width * height * 3);

    for (unsigned y = 0; y < height; y++) {
      TIFFReadScanline(data, &pixels[width * 3 * y], y);
    }

    TIFFClose(data);

    return NewTexture2d(width, height, GL_RGB32F, GL_RGB, GL_FLOAT, pixels.data());
  };

  mPhaseTexture         = read2D(settings.mPhaseTexture);
  mTransmittanceTexture = read2D(settings.mTransmittanceTexture);
  mIrradianceTexture    = read2D(settings.mIrradianceTexture);

  auto read3D = [](std::string const& path) {
    auto* data = TIFFOpen(path.c_str(), "r");

    if (!data) {
      logger().error("Failed to open TIFF file '{}'", path);
      return 0u;
    }

    uint32_t width{};
    uint32_t height{};
    uint32_t depth{};

    TIFFGetField(data, TIFFTAG_IMAGELENGTH, &height);
    TIFFGetField(data, TIFFTAG_IMAGEWIDTH, &width);

    do {
      depth++;
    } while (TIFFReadDirectory(data));

    std::vector<float> pixels(width * height * depth * 3);

    for (unsigned z = 0; z < depth; z++) {
      TIFFSetDirectory(data, z);
      for (unsigned y = 0; y < height; y++) {
        TIFFReadScanline(data, &pixels[width * 3 * y + (3 * width * height * z)], y);
      }
    }

    TIFFClose(data);

    return NewTexture3d(width, height, depth, GL_RGB32F, GL_RGB, GL_FLOAT, pixels.data());
  };

  mMultipleScatteringTexture       = read3D(settings.mMultipleScatteringTexture);
  mSingleAerosolsScatteringTexture = read3D(settings.mSingleScatteringTexture);

  // Create and compile the shader providing our API.
  const char* source = shader.c_str();
  mAtmosphereShader  = glCreateShader(GL_FRAGMENT_SHADER);
  glShaderSource(mAtmosphereShader, 1, &source, NULL);
  glCompileShader(mAtmosphereShader);

  return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

GLuint Model::getShader() const {
  return mAtmosphereShader;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

GLuint Model::setUniforms(GLuint program, GLuint startTextureUnit) const {

  glActiveTexture(GL_TEXTURE0 + startTextureUnit + 0);
  glBindTexture(GL_TEXTURE_2D, mPhaseTexture);
  glUniform1i(glGetUniformLocation(program, "uPhaseTexture"), startTextureUnit + 0);

  glActiveTexture(GL_TEXTURE0 + startTextureUnit + 1);
  glBindTexture(GL_TEXTURE_2D, mTransmittanceTexture);
  glUniform1i(glGetUniformLocation(program, "uTransmittanceTexture"), startTextureUnit + 1);

  glActiveTexture(GL_TEXTURE0 + startTextureUnit + 2);
  glBindTexture(GL_TEXTURE_3D, mMultipleScatteringTexture);
  glUniform1i(glGetUniformLocation(program, "uMultipleScatteringTexture"), startTextureUnit + 2);

  glActiveTexture(GL_TEXTURE0 + startTextureUnit + 3);
  glBindTexture(GL_TEXTURE_2D, mIrradianceTexture);
  glUniform1i(glGetUniformLocation(program, "uIrradianceTexture"), startTextureUnit + 3);

  glActiveTexture(GL_TEXTURE0 + startTextureUnit + 4);
  glBindTexture(GL_TEXTURE_3D, mSingleAerosolsScatteringTexture);
  glUniform1i(
      glGetUniformLocation(program, "uSingleAerosolsScatteringTexture"), startTextureUnit + 4);

  return startTextureUnit + 5;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::atmospheres::models::bruneton
