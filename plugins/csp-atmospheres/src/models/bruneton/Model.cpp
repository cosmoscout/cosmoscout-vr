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
#include "Metadata.hpp"

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

void from_json(nlohmann::json const& j, Model::Settings& o) {
  cs::core::Settings::deserialize(j, "dataDirectory", o.mDataDirectory);
}

void to_json(nlohmann::json& j, Model::Settings const& o) {
  cs::core::Settings::serialize(j, "dataDirectory", o.mDataDirectory);
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

  Metadata meta;

  try {
    std::ifstream  file(settings.mDataDirectory + "/metadata.json");
    nlohmann::json j;
    file >> j;
    meta = j;
  } catch (std::exception const& e) {
    logger().error("Failed to parse atmosphere parameters: {}", e.what());
  }

  // A lambda that creates a GLSL header containing our atmosphere computation functions,
  // specialized for the given atmosphere parameters and for the 3 wavelengths in 'lambdas'.
  auto model = cs::utils::filesystem::loadToString(
      "../share/resources/shaders/atmosphere-models/bruneton/model.glsl");
  auto common = cs::utils::filesystem::loadToString(
      "../share/resources/shaders/atmosphere-models/bruneton/common.glsl");

  mPhaseTexture = std::get<0>(read2DTexture(settings.mDataDirectory + "/phase.tif"));

  {
    auto const [t, w, h]        = read2DTexture(settings.mDataDirectory + "/transmittance.tif");
    mTransmittanceTexture       = t;
    mTransmittanceTextureWidth  = w;
    mTransmittanceTextureHeight = h;
  }

  {
    auto const [t, w, h]     = read2DTexture(settings.mDataDirectory + "/indirect_illuminance.tif");
    mIrradianceTexture       = t;
    mIrradianceTextureWidth  = w;
    mIrradianceTextureHeight = h;
  }

  {
    auto const [t, w, h, d] = read3DTexture(settings.mDataDirectory + "/multiple_scattering.tif");
    mMultipleScatteringTexture = t;
    mScatteringTextureNuSize   = meta.mScatteringTextureNuSize;
    mScatteringTextureMuSSize  = w / mScatteringTextureNuSize;
    mScatteringTextureMuSize   = h;
    mScatteringTextureRSize    = d;
  }

  mSingleAerosolsScatteringTexture =
      std::get<0>(read3DTexture(settings.mDataDirectory + "/single_aerosols_scattering.tif"));

  // clang-format off
  std::string shader =
    std::string("#version 330\n") +
    "const int TRANSMITTANCE_TEXTURE_WIDTH = "  + cs::utils::toString(mTransmittanceTextureWidth) + ";\n" +
    "const int TRANSMITTANCE_TEXTURE_HEIGHT = " + cs::utils::toString(mTransmittanceTextureHeight) + ";\n" +
    "const int SCATTERING_TEXTURE_R_SIZE = "    + cs::utils::toString(mScatteringTextureRSize) + ";\n" +
    "const int SCATTERING_TEXTURE_MU_SIZE = "   + cs::utils::toString(mScatteringTextureMuSize) + ";\n" +
    "const int SCATTERING_TEXTURE_MU_S_SIZE = " + cs::utils::toString(mScatteringTextureMuSSize) + ";\n" +
    "const int SCATTERING_TEXTURE_NU_SIZE = "   + cs::utils::toString(mScatteringTextureNuSize) + ";\n" +
    "const int IRRADIANCE_TEXTURE_WIDTH = "     + cs::utils::toString(mIrradianceTextureWidth) + ";\n" +
    "const int IRRADIANCE_TEXTURE_HEIGHT = "    + cs::utils::toString(mIrradianceTextureHeight) + ";\n" +
    "const vec3 SOLAR_ILLUMINANCE = vec3("      + cs::utils::toString(meta.mSunIlluminance.r) + "," + cs::utils::toString(meta.mSunIlluminance.g) + "," + cs::utils::toString(meta.mSunIlluminance.b) + ");\n" +
    "const float SUN_ANGULAR_RADIUS = "         + cs::utils::toString(meta.mSunAngularRadius) + ";\n" +
    "const float BOTTOM_RADIUS = "              + cs::utils::toString(planetRadius) + ";\n" +
    "const float TOP_RADIUS = "                 + cs::utils::toString(atmosphereRadius) + ";\n" +
    "const float MU_S_MIN = "                   + cs::utils::toString(std::cos(meta.mMaxSunZenithAngle))+ ";\n" +
    common + "\n" +
    model;
  // clang-format on

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

std::tuple<GLuint, int32_t, int32_t> Model::read2DTexture(std::string const& path) const {
  auto* data = TIFFOpen(path.c_str(), "r");

  if (!data) {
    logger().error("Failed to open TIFF file '{}'", path);
    return {0u, 0, 0};
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

  GLuint texture;
  glGenTextures(1, &texture);
  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, texture);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, width, height, 0, GL_RGB, GL_FLOAT, pixels.data());

  return {texture, width, height};
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::tuple<GLuint, int32_t, int32_t, int32_t> Model::read3DTexture(std::string const& path) const {
  auto* data = TIFFOpen(path.c_str(), "r");

  if (!data) {
    logger().error("Failed to open TIFF file '{}'", path);
    return {0u, 0, 0, 0};
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
  glTexImage3D(
      GL_TEXTURE_3D, 0, GL_RGB32F, width, height, depth, 0, GL_RGB, GL_FLOAT, pixels.data());

  return {texture, width, height, depth};
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::atmospheres::models::bruneton
