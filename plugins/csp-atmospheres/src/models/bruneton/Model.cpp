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

} // namespace

////////////////////////////////////////////////////////////////////////////////////////////////////

void from_json(nlohmann::json const& j, Model::Settings& o) {
  cs::core::Settings::deserialize(j, "sunAngularRadius", o.mSunAngularRadius);
  cs::core::Settings::deserialize(j, "sunIlluminance", o.mSunIlluminance);
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
  cs::core::Settings::serialize(j, "sunIlluminance", o.mSunIlluminance);
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
    "const vec3 SOLAR_ILLUMINANCE = vec3(" + cs::utils::toString(settings.mSunIlluminance.r) + "," + cs::utils::toString(settings.mSunIlluminance.g) + "," + cs::utils::toString(settings.mSunIlluminance.b) + ");\n" +
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
