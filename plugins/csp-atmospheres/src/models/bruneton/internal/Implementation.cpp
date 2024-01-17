////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-FileCopyrightText: 2017 Eric Bruneton
// SPDX-License-Identifier: BSD-3-Clause

#include "Implementation.hpp"

#include "../../../logger.hpp"
#include "CSVLoader.hpp"

#include "../../../../src/cs-utils/filesystem.hpp"
#include "../../../../src/cs-utils/utils.hpp"

#include <cassert>
#include <cmath>
#include <glm/gtc/type_ptr.hpp>
#include <iostream>
#include <memory>

// This file is based in large parts on the original implementation by Eric Bruneton:
// https://github.com/ebruneton/precomputed_atmospheric_scattering/blob/master/atmosphere/model.cc

// While implementing the atmospheric model into CosmoScout VR, we have refactored some parts of the
// code, however this is mostly related to how variables are named and how input parameters are
// passed to the model. The only fundamental change is that the phase functions for aerosols and
// molecules as well as their density distributions are now loaded from CSV files and then later
// sampled from textures.

// Below, we will indicate for each group of function whether something has been changed and a link
// to the original explanations of the methods by Eric Bruneton.

namespace csp::atmospheres::models::bruneton::internal {

namespace {

// Values from "Reference Solar Spectral Irradiance: ASTM G-173", ETR column  (see
// http://rredc.nrel.gov/solar/spectra/am1.5/ASTMG173/ASTMG173.html), summed and averaged in each
// bin (e.g. the value for 360nm is the average of the ASTM G-173 values for all wavelengths between
// 360 and 370nm). Values in W.m^-2. Copied from:
// https://github.com/ebruneton/precomputed_atmospheric_scattering/blob/master/atmosphere/demo/demo.cc
// clang-format off
const std::vector<double> SOLAR_IRRADIANCE = {
                                                          1.11776, 1.14259, 1.01249, 1.14716,
    1.72765, 1.73054, 1.6887,  1.61253, 1.91198, 2.03474, 2.02042, 2.02212, 1.93377, 1.95809,
    1.91686, 1.8298,  1.8685,  1.8931,  1.85149, 1.8504,  1.8341,  1.8345,  1.8147,  1.78158,
    1.7533,  1.6965,  1.68194, 1.64654, 1.6048,  1.52143, 1.55622, 1.5113,  1.474,   1.4482,
    1.41018, 1.36775, 1.34188, 1.31429, 1.28303, 1.26758, 1.2367,  1.2082,  1.18737, 1.14683,
    1.12362, 1.1058, 1.07124, 1.04992
};

const std::vector<double> WAVELENGTHS = {
                                  360, 370, 380, 390,
    400, 410, 420, 430, 440, 450, 460, 470, 480, 490,
    500, 510, 520, 530, 540, 550, 560, 570, 580, 590,
    600, 610, 620, 630, 640, 650, 660, 670, 680, 690,
    700, 710, 720, 730, 740, 750, 760, 770, 780, 790,
    800, 810, 820, 830
};
// clang-format on

// The conversion factor between watts and lumens.
constexpr double MAX_LUMINOUS_EFFICACY = 683.0;

// Values from "CIE (1931) 2-deg color matching functions", see
// "http://web.archive.org/web/20081228084047/http://www.cvrl.org/database/data/cmfs/ciexyz31.txt".
// Copied from:
// https://github.com/ebruneton/precomputed_atmospheric_scattering/blob/master/atmosphere/constants.h
// clang-format off
constexpr double CIE_2_DEG_COLOR_MATCHING_FUNCTIONS[380] = {
    360, 0.000129900000, 0.000003917000, 0.000606100000,
    365, 0.000232100000, 0.000006965000, 0.001086000000,
    370, 0.000414900000, 0.000012390000, 0.001946000000,
    375, 0.000741600000, 0.000022020000, 0.003486000000,
    380, 0.001368000000, 0.000039000000, 0.006450001000,
    385, 0.002236000000, 0.000064000000, 0.010549990000,
    390, 0.004243000000, 0.000120000000, 0.020050010000,
    395, 0.007650000000, 0.000217000000, 0.036210000000,
    400, 0.014310000000, 0.000396000000, 0.067850010000,
    405, 0.023190000000, 0.000640000000, 0.110200000000,
    410, 0.043510000000, 0.001210000000, 0.207400000000,
    415, 0.077630000000, 0.002180000000, 0.371300000000,
    420, 0.134380000000, 0.004000000000, 0.645600000000,
    425, 0.214770000000, 0.007300000000, 1.039050100000,
    430, 0.283900000000, 0.011600000000, 1.385600000000,
    435, 0.328500000000, 0.016840000000, 1.622960000000,
    440, 0.348280000000, 0.023000000000, 1.747060000000,
    445, 0.348060000000, 0.029800000000, 1.782600000000,
    450, 0.336200000000, 0.038000000000, 1.772110000000,
    455, 0.318700000000, 0.048000000000, 1.744100000000,
    460, 0.290800000000, 0.060000000000, 1.669200000000,
    465, 0.251100000000, 0.073900000000, 1.528100000000,
    470, 0.195360000000, 0.090980000000, 1.287640000000,
    475, 0.142100000000, 0.112600000000, 1.041900000000,
    480, 0.095640000000, 0.139020000000, 0.812950100000,
    485, 0.057950010000, 0.169300000000, 0.616200000000,
    490, 0.032010000000, 0.208020000000, 0.465180000000,
    495, 0.014700000000, 0.258600000000, 0.353300000000,
    500, 0.004900000000, 0.323000000000, 0.272000000000,
    505, 0.002400000000, 0.407300000000, 0.212300000000,
    510, 0.009300000000, 0.503000000000, 0.158200000000,
    515, 0.029100000000, 0.608200000000, 0.111700000000,
    520, 0.063270000000, 0.710000000000, 0.078249990000,
    525, 0.109600000000, 0.793200000000, 0.057250010000,
    530, 0.165500000000, 0.862000000000, 0.042160000000,
    535, 0.225749900000, 0.914850100000, 0.029840000000,
    540, 0.290400000000, 0.954000000000, 0.020300000000,
    545, 0.359700000000, 0.980300000000, 0.013400000000,
    550, 0.433449900000, 0.994950100000, 0.008749999000,
    555, 0.512050100000, 1.000000000000, 0.005749999000,
    560, 0.594500000000, 0.995000000000, 0.003900000000,
    565, 0.678400000000, 0.978600000000, 0.002749999000,
    570, 0.762100000000, 0.952000000000, 0.002100000000,
    575, 0.842500000000, 0.915400000000, 0.001800000000,
    580, 0.916300000000, 0.870000000000, 0.001650001000,
    585, 0.978600000000, 0.816300000000, 0.001400000000,
    590, 1.026300000000, 0.757000000000, 0.001100000000,
    595, 1.056700000000, 0.694900000000, 0.001000000000,
    600, 1.062200000000, 0.631000000000, 0.000800000000,
    605, 1.045600000000, 0.566800000000, 0.000600000000,
    610, 1.002600000000, 0.503000000000, 0.000340000000,
    615, 0.938400000000, 0.441200000000, 0.000240000000,
    620, 0.854449900000, 0.381000000000, 0.000190000000,
    625, 0.751400000000, 0.321000000000, 0.000100000000,
    630, 0.642400000000, 0.265000000000, 0.000049999990,
    635, 0.541900000000, 0.217000000000, 0.000030000000,
    640, 0.447900000000, 0.175000000000, 0.000020000000,
    645, 0.360800000000, 0.138200000000, 0.000010000000,
    650, 0.283500000000, 0.107000000000, 0.000000000000,
    655, 0.218700000000, 0.081600000000, 0.000000000000,
    660, 0.164900000000, 0.061000000000, 0.000000000000,
    665, 0.121200000000, 0.044580000000, 0.000000000000,
    670, 0.087400000000, 0.032000000000, 0.000000000000,
    675, 0.063600000000, 0.023200000000, 0.000000000000,
    680, 0.046770000000, 0.017000000000, 0.000000000000,
    685, 0.032900000000, 0.011920000000, 0.000000000000,
    690, 0.022700000000, 0.008210000000, 0.000000000000,
    695, 0.015840000000, 0.005723000000, 0.000000000000,
    700, 0.011359160000, 0.004102000000, 0.000000000000,
    705, 0.008110916000, 0.002929000000, 0.000000000000,
    710, 0.005790346000, 0.002091000000, 0.000000000000,
    715, 0.004109457000, 0.001484000000, 0.000000000000,
    720, 0.002899327000, 0.001047000000, 0.000000000000,
    725, 0.002049190000, 0.000740000000, 0.000000000000,
    730, 0.001439971000, 0.000520000000, 0.000000000000,
    735, 0.000999949300, 0.000361100000, 0.000000000000,
    740, 0.000690078600, 0.000249200000, 0.000000000000,
    745, 0.000476021300, 0.000171900000, 0.000000000000,
    750, 0.000332301100, 0.000120000000, 0.000000000000,
    755, 0.000234826100, 0.000084800000, 0.000000000000,
    760, 0.000166150500, 0.000060000000, 0.000000000000,
    765, 0.000117413000, 0.000042400000, 0.000000000000,
    770, 0.000083075270, 0.000030000000, 0.000000000000,
    775, 0.000058706520, 0.000021200000, 0.000000000000,
    780, 0.000041509940, 0.000014990000, 0.000000000000,
    785, 0.000029353260, 0.000010600000, 0.000000000000,
    790, 0.000020673830, 0.000007465700, 0.000000000000,
    795, 0.000014559770, 0.000005257800, 0.000000000000,
    800, 0.000010253980, 0.000003702900, 0.000000000000,
    805, 0.000007221456, 0.000002607800, 0.000000000000,
    810, 0.000005085868, 0.000001836600, 0.000000000000,
    815, 0.000003581652, 0.000001293400, 0.000000000000,
    820, 0.000002522525, 0.000000910930, 0.000000000000,
    825, 0.000001776509, 0.000000641530, 0.000000000000,
    830, 0.000001251141, 0.000000451810, 0.000000000000,
};
// clang-format on

// The conversion matrix from XYZ to linear sRGB color spaces.
// Values from https://en.wikipedia.org/wiki/SRGB.
// clang-format off
constexpr double XYZ_TO_SRGB[9] = {
    +3.2406, -1.5372, -0.4986,
    -0.9689, +1.8758, +0.0415,
    +0.0557, -0.2040, +1.0570
};
// clang-format on

// Shader Definitions ------------------------------------------------------------------------------

// Below, the source code for several shaders is defined. Most of them are used during the
// pre-processing. Only the last one is is linked into the final fragment shader used at run-time.

// An explanation of the following shaders is available online:
// https://ebruneton.github.io/precomputed_atmospheric_scattering/atmosphere/model.cc.html#shaders

// The only functional difference is that the kAtmosphereShader does not provide the radiance API
// anymore as it is not required by CosmoScout VR. Also, the shadow_length parameters have been
// removed and the GetSunAndSkyIlluminance() does not require the surface normal anymore.

const char kVertexShader[] = R"(
  #version 330

  layout(location = 0) in vec2 iVertex;

  void main() {
    gl_Position = vec4(iVertex, 0.0, 1.0);
  }
)";

const char kGeometryShader[] = R"(
  #version 330

  layout(triangles) in;
  layout(triangle_strip, max_vertices = 3) out;

  uniform int uLayer;

  void main() {
    gl_Position = gl_in[0].gl_Position;
    gl_Layer = uLayer;
    EmitVertex();

    gl_Position = gl_in[1].gl_Position;
    gl_Layer = uLayer;
    EmitVertex();

    gl_Position = gl_in[2].gl_Position;
    gl_Layer = uLayer;
    EmitVertex();

    EndPrimitive();
  }
)";

const char kComputeTransmittanceShader[] = R"(
  layout(location = 0) out vec3 oTransmittance;

  void main() {
    oTransmittance = computeTransmittanceToTopAtmosphereBoundaryTexture(ATMOSPHERE, gl_FragCoord.xy);
  }
)";

const char kComputeDirectIrradianceShader[] = R"(
  layout(location = 0) out vec3 oDeltaIrradiance;
  layout(location = 1) out vec3 oIrradiance;
  uniform sampler2D uTransmittanceTexture;
  void main() {
    oDeltaIrradiance = computeDirectIrradianceTexture(uTransmittanceTexture, gl_FragCoord.xy);
    oIrradiance = vec3(0.0);
  }
)";

const char kComputeSingleScatteringShader[] = R"(
  layout(location = 0) out vec3 oDeltaMolecules;
  layout(location = 1) out vec3 oDeltaAerosols;
  layout(location = 2) out vec3 oAccumulatedMoleculesSingleScatteringLuminance;
  layout(location = 3) out vec3 oAccumulatedAerosolsSingleScatteringLuminance;

  uniform mat3 uLuminanceFromRadiance;
  uniform sampler2D uTransmittanceTexture;
  uniform int uLayer;

  void main() {
    computeSingleScatteringTexture(ATMOSPHERE, uTransmittanceTexture,
                                   vec3(gl_FragCoord.xy, uLayer + 0.5),
                                   oDeltaMolecules, oDeltaAerosols);
    oAccumulatedMoleculesSingleScatteringLuminance = uLuminanceFromRadiance * oDeltaMolecules;
    oAccumulatedAerosolsSingleScatteringLuminance = uLuminanceFromRadiance * oDeltaAerosols;
  }
)";

const char kComputeScatteringDensityShader[] = R"(
  layout(location = 0) out vec3 oScatteringDensity;

  uniform sampler2D uTransmittanceTexture;
  uniform sampler3D uSingleMoleculesScatteringTexture;
  uniform sampler3D uSingleAerosolsScatteringTexture;
  uniform sampler3D uMultipleScatteringTexture;
  uniform sampler2D uIrradianceTexture;
  uniform int uScatteringOrder;
  uniform int uLayer;

  void main() {
    oScatteringDensity = computeScatteringDensityTexture(ATMOSPHERE, uTransmittanceTexture,
                                                         uSingleMoleculesScatteringTexture,
                                                         uSingleAerosolsScatteringTexture,
                                                         uMultipleScatteringTexture,
                                                         uIrradianceTexture,
                                                         vec3(gl_FragCoord.xy, uLayer + 0.5),
                                                         uScatteringOrder);
  }
)";

const char kComputeIndirectIrradianceShader[] = R"(
  layout(location = 0) out vec3 oDeltaIrradiance;
  layout(location = 1) out vec3 oIrradiance;

  uniform mat3 uLuminanceFromRadiance;
  uniform sampler3D uSingleMoleculesScatteringTexture;
  uniform sampler3D uSingleAerosolsScatteringTexture;
  uniform sampler3D uMultipleScatteringTexture;
  uniform int uScatteringOrder;

  void main() {
    oDeltaIrradiance = computeIndirectIrradianceTexture(ATMOSPHERE, uSingleMoleculesScatteringTexture,
                                                        uSingleAerosolsScatteringTexture,
                                                        uMultipleScatteringTexture,
                                                        gl_FragCoord.xy, uScatteringOrder);
    oIrradiance = uLuminanceFromRadiance * oDeltaIrradiance;
  }
)";

const char kComputeMultipleScatteringShader[] = R"(
  layout(location = 0) out vec3 oDeltaMultipleScattering;
  layout(location = 1) out vec3 oMultipleScattering;

  uniform mat3 uLuminanceFromRadiance;
  uniform sampler2D uTransmittanceTexture;
  uniform sampler3D uScatteringDensityTexture;
  uniform int uLayer;
  
  void main() {
    float nu;
    oDeltaMultipleScattering = computeMultipleScatteringTexture(uTransmittanceTexture,
                                                                uScatteringDensityTexture,
                                                                vec3(gl_FragCoord.xy, uLayer + 0.5),
                                                                nu);
    oMultipleScattering = uLuminanceFromRadiance * oDeltaMultipleScattering /
                         phaseFunction(ATMOSPHERE.molecules, nu);
  }
)";

const char kAtmosphereShader[] = R"(
  uniform sampler2D uTransmittanceTexture;
  uniform sampler3D uMultipleScatteringTexture;
  uniform sampler3D uSingleAerosolsScatteringTexture;
  uniform sampler2D uIrradianceTexture;

  vec3 GetSkyLuminance(vec3 camera, vec3 viewRay, vec3 sunDirection, out vec3 transmittance) {
    return getSkyRadiance(ATMOSPHERE, uTransmittanceTexture, uMultipleScatteringTexture,
                          uSingleAerosolsScatteringTexture, camera, viewRay,
                          sunDirection, transmittance) * SKY_SPECTRAL_RADIANCE_TO_LUMINANCE;
  }

  vec3 GetSkyLuminanceToPoint(vec3 camera, vec3 point, vec3 sunDirection, out vec3 transmittance) {
    return getSkyRadianceToPoint(ATMOSPHERE, uTransmittanceTexture, uMultipleScatteringTexture,
                                 uSingleAerosolsScatteringTexture, camera, point,
                                 sunDirection, transmittance) * SKY_SPECTRAL_RADIANCE_TO_LUMINANCE;
  }

  vec3 GetSunAndSkyIlluminance(vec3 p, vec3 sunDirection, out vec3 skyIrradiance) {
    vec3 sun_irradiance = getSunAndSkyIrradiance(uTransmittanceTexture, uIrradianceTexture, p,
                                                 sunDirection, skyIrradiance);
    skyIrradiance *= SKY_SPECTRAL_RADIANCE_TO_LUMINANCE;
    return sun_irradiance * SUN_SPECTRAL_RADIANCE_TO_LUMINANCE;
  }
)";

// Utility classes and Functions -------------------------------------------------------------------

// The functions and classes below are local to this file and used further down in the actual model
// implementation.

// An explanation of the classes and methods is available online:
// https://ebruneton.github.io/precomputed_atmospheric_scattering/atmosphere/model.cc.html#utilities

// This shader program class has not been changed functionality-wise.

class Program {
 public:
  Program(std::string const& vertexShaderSource, std::string const& fragmentShaderSource)
      : Program(vertexShaderSource, "", fragmentShaderSource) {
  }

  Program(std::string const& vertexShaderSource, std::string const& geometryShaderSource,
      std::string const& fragmentShaderSource) {
    mProgram = glCreateProgram();

    const char* source;
    source              = vertexShaderSource.c_str();
    GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &source, NULL);
    glCompileShader(vertexShader);
    CheckShader(vertexShader);
    glAttachShader(mProgram, vertexShader);

    GLuint geometryShader = 0;
    if (!geometryShaderSource.empty()) {
      source         = geometryShaderSource.c_str();
      geometryShader = glCreateShader(GL_GEOMETRY_SHADER);
      glShaderSource(geometryShader, 1, &source, NULL);
      glCompileShader(geometryShader);
      CheckShader(geometryShader);
      glAttachShader(mProgram, geometryShader);
    }

    source                = fragmentShaderSource.c_str();
    GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &source, NULL);
    glCompileShader(fragmentShader);
    CheckShader(fragmentShader);
    glAttachShader(mProgram, fragmentShader);

    glLinkProgram(mProgram);
    CheckProgram(mProgram);

    glDetachShader(mProgram, vertexShader);
    glDeleteShader(vertexShader);
    if (!geometryShaderSource.empty()) {
      glDetachShader(mProgram, geometryShader);
      glDeleteShader(geometryShader);
    }
    glDetachShader(mProgram, fragmentShader);
    glDeleteShader(fragmentShader);
  }

  ~Program() {
    glDeleteProgram(mProgram);
  }

  void Use() const {
    glUseProgram(mProgram);
  }

  void BindMat3(std::string const& uniformName, glm::mat3 const& value) const {
    glUniformMatrix3fv(glGetUniformLocation(mProgram, uniformName.c_str()), 1, true /* transpose */,
        glm::value_ptr(value));
  }

  void BindInt(std::string const& uniformName, int value) const {
    glUniform1i(glGetUniformLocation(mProgram, uniformName.c_str()), value);
  }

  void BindTexture2d(
      std::string const& samplerUniformName, GLuint texture, GLuint textureUnit) const {
    glActiveTexture(GL_TEXTURE0 + textureUnit);
    glBindTexture(GL_TEXTURE_2D, texture);
    BindInt(samplerUniformName, textureUnit);
  }

  void BindTexture3d(
      std::string const& samplerUniformName, GLuint texture, GLuint textureUnit) const {
    glActiveTexture(GL_TEXTURE0 + textureUnit);
    glBindTexture(GL_TEXTURE_3D, texture);
    BindInt(samplerUniformName, textureUnit);
  }

 private:
  static void CheckShader(GLuint shader) {
    GLint compileStatus;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &compileStatus);
    if (compileStatus == GL_FALSE) {
      PrintShaderLog(shader);
    }
    assert(compileStatus == GL_TRUE);
  }

  static void PrintShaderLog(GLuint shader) {
    GLint logLength;
    glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &logLength);
    if (logLength > 0) {
      std::unique_ptr<char[]> log_data(new char[logLength]);
      glGetShaderInfoLog(shader, logLength, &logLength, log_data.get());
      std::cerr << "compile log = " << std::string(log_data.get(), logLength) << std::endl;
    }
  }

  static void CheckProgram(GLuint program) {
    GLint linkStatus;
    glGetProgramiv(program, GL_LINK_STATUS, &linkStatus);
    if (linkStatus == GL_FALSE) {
      PrintProgramLog(program);
    }
    assert(linkStatus == GL_TRUE);
    assert(glGetError() == 0);
  }

  static void PrintProgramLog(GLuint program) {
    GLint logLength;
    glGetProgramiv(program, GL_INFO_LOG_LENGTH, &logLength);
    if (logLength > 0) {
      std::unique_ptr<char[]> log_data(new char[logLength]);
      glGetProgramInfoLog(program, logLength, &logLength, log_data.get());
      std::cerr << "link log = " << std::string(log_data.get(), logLength) << std::endl;
    }
  }

  GLuint mProgram;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// These GL-texture generators have been made a bit more flexible to allow for different pixel
// formats. At the same time, we have removed support for the half-resolution pixel formats (they
// quickly lead to artefacts with the high-dynamic range of CosmoScout VR). Also, we do not check
// for the availability of RGB textures anymore but use RGBA textures everywhere.

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

void DrawQuad(std::vector<bool> const& enableBlend, GLuint quadVAO) {
  for (unsigned int i = 0; i < enableBlend.size(); ++i) {
    if (enableBlend[i]) {
      glEnablei(GL_BLEND, i);
    }
  }

  glBindVertexArray(quadVAO);
  glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
  glBindVertexArray(0);

  for (unsigned int i = 0; i < enableBlend.size(); ++i) {
    glDisablei(GL_BLEND, i);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// This is functionality-wise identical to the original implementation.

double CieColorMatchingFunctionTableValue(double wavelength, int column) {
  if (wavelength <= WAVELENGTHS.front() || wavelength >= WAVELENGTHS.back()) {
    return 0.0;
  }
  double u   = (wavelength - WAVELENGTHS.front()) / 5.0;
  int    row = static_cast<int>(std::floor(u));
  assert(row >= 0 && row + 1 < 95);
  assert(CIE_2_DEG_COLOR_MATCHING_FUNCTIONS[4 * row] <= wavelength &&
         CIE_2_DEG_COLOR_MATCHING_FUNCTIONS[4 * (row + 1)] >= wavelength);
  u -= row;
  return CIE_2_DEG_COLOR_MATCHING_FUNCTIONS[4 * row + column] * (1.0 - u) +
         CIE_2_DEG_COLOR_MATCHING_FUNCTIONS[4 * (row + 1) + column] * u;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// This is functionality-wise identical to the original implementation.

double Interpolate(std::vector<double> const& xVals, std::vector<double> const& yVals, double x) {
  assert(yVals.size() == xVals.size());

  if (x < xVals[0]) {
    return yVals[0];
  }

  for (unsigned int i = 0; i < xVals.size() - 1; ++i) {
    if (x < xVals[i + 1]) {
      double u = (x - xVals[i]) / (xVals[i + 1] - xVals[i]);
      return yVals[i] * (1.0 - u) + yVals[i + 1] * u;
    }
  }

  return yVals[yVals.size() - 1];
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// This is functionality-wise identical to the original implementation.

void ComputeSpectralRadianceToLuminanceFactors(
    double lambdaPower, double* kR, double* kG, double* kB) {

  *kR            = 0.0;
  *kG            = 0.0;
  *kB            = 0.0;
  double solarR  = Interpolate(WAVELENGTHS, SOLAR_IRRADIANCE, Implementation::kLambdaR);
  double solarG  = Interpolate(WAVELENGTHS, SOLAR_IRRADIANCE, Implementation::kLambdaG);
  double solarB  = Interpolate(WAVELENGTHS, SOLAR_IRRADIANCE, Implementation::kLambdaB);
  int    dLambda = 1;
  for (int lambda = WAVELENGTHS.front(); lambda <= WAVELENGTHS.back(); lambda += dLambda) {
    double        x_bar      = CieColorMatchingFunctionTableValue(lambda, 1);
    double        y_bar      = CieColorMatchingFunctionTableValue(lambda, 2);
    double        z_bar      = CieColorMatchingFunctionTableValue(lambda, 3);
    const double* xyz2srgb   = XYZ_TO_SRGB;
    double        r_bar      = xyz2srgb[0] * x_bar + xyz2srgb[1] * y_bar + xyz2srgb[2] * z_bar;
    double        g_bar      = xyz2srgb[3] * x_bar + xyz2srgb[4] * y_bar + xyz2srgb[5] * z_bar;
    double        b_bar      = xyz2srgb[6] * x_bar + xyz2srgb[7] * y_bar + xyz2srgb[8] * z_bar;
    double        irradiance = Interpolate(WAVELENGTHS, SOLAR_IRRADIANCE, lambda);
    *kR += r_bar * irradiance / solarR * pow(lambda / Implementation::kLambdaR, lambdaPower);
    *kG += g_bar * irradiance / solarG * pow(lambda / Implementation::kLambdaG, lambdaPower);
    *kB += b_bar * irradiance / solarB * pow(lambda / Implementation::kLambdaB, lambdaPower);
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
    std::vector<double> const& xVals, std::vector<double> const& yVals, glm::dvec3 const& lambdas) {
  double r = Interpolate(xVals, yVals, lambdas[0]);
  double g = Interpolate(xVals, yVals, lambdas[1]);
  double b = Interpolate(xVals, yVals, lambdas[2]);
  return "vec3(" + cs::utils::toString(r) + "," + cs::utils::toString(g) + "," +
         cs::utils::toString(b) + ")";
}

// This creates an GLSL snippet corresponding to the given scattering component.
std::string printScatteringComponent(Params::ScatteringComponent const& component,
    std::vector<double> const& wavelengths, float phaseTextureV, float densityTextureV,
    glm::dvec3 const& lambdas) {

  auto absorption = extractVec3(wavelengths, component.mAbsorption, lambdas);
  auto scattering = extractVec3(wavelengths, component.mScattering, lambdas);

  std::stringstream ss;
  ss << "ScatteringComponent(";
  ss << phaseTextureV << ",\n";
  ss << densityTextureV << ",\n";
  ss << scattering << " + " << absorption << ",\n";
  ss << scattering << "\n";
  ss << ")";

  return ss.str();
}

// This creates an GLSL snippet corresponding to the given absorbing component.
std::string printAbsorbingComponent(Params::AbsorbingComponent const& component,
    std::vector<double> const& wavelengths, float densityTextureV, glm::dvec3 const& lambdas) {

  auto absorption = extractVec3(wavelengths, component.mAbsorption, lambdas);

  std::stringstream ss;
  ss << "AbsorbingComponent(";
  ss << densityTextureV << ",\n";
  ss << absorption << "\n";
  ss << ")";

  return ss.str();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // anonymous namespace

////////////////////////////////////////////////////////////////////////////////////////////////////

// Model Implementation ---------------------------------------------------------------------------

// The code below roughly follows the original implementation by Eric Bruneton.

// The original explanation of the methods still applies in most parts and is available online:
// https://ebruneton.github.io/precomputed_atmospheric_scattering/atmosphere/model.cc.html#implementation

// The main differences in the constructor are that we pass significantly more constants to the
// shader code as all the sampling counts are configurable now. Also, we create the new density
// texture which contains the density profiles for all atmosphere constituents.
Implementation::Implementation(Params params)
    : mParams(std::move(params))
    , mScatteringTextureWidth(mParams.mScatteringTextureNuSize * mParams.mScatteringTextureMuSSize)
    , mScatteringTextureHeight(mParams.mScatteringTextureMuSize)
    , mScatteringTextureDepth(mParams.mScatteringTextureRSize) {

  // Compute the values for the SKY_RADIANCE_TO_LUMINANCE constant. In theory this should be 1 in
  // precomputed illuminance mode (because the precomputed textures already contain illuminance
  // values). In practice, however, storing true illuminance values in half precision textures
  // yields artefacts (because the values are too large), so we store illuminance values divided by
  // MAX_LUMINOUS_EFFICACY instead. This is why, in precomputed illuminance mode, we set
  // SKY_RADIANCE_TO_LUMINANCE to MAX_LUMINOUS_EFFICACY.
  bool   precomputeIlluminance = mParams.mWavelengths.size() > 3;
  double skyKR, skyKG, skyKB;
  if (precomputeIlluminance) {
    skyKR = skyKG = skyKB = MAX_LUMINOUS_EFFICACY;
  } else {
    ComputeSpectralRadianceToLuminanceFactors(-3 /* lambdaPower */, &skyKR, &skyKG, &skyKB);
  }
  // Compute the values for the SUN_RADIANCE_TO_LUMINANCE constant.
  double sunKR, sunKG, sunKB;
  ComputeSpectralRadianceToLuminanceFactors(0 /* lambdaPower */, &sunKR, &sunKG, &sunKB);

  // A lambda that creates a GLSL header containing our atmosphere computation functions,
  // specialized for the given atmosphere parameters and for the 3 wavelengths in 'lambdas'.
  auto definitions = cs::utils::filesystem::loadToString(
      "../share/resources/shaders/csp-atmospheres/models/bruneton/definitions.glsl");
  auto functions = cs::utils::filesystem::loadToString(
      "../share/resources/shaders/csp-atmospheres/models/bruneton/functions.glsl");

  // clang-format off
  mGlslHeaderFactory = [=](glm::dvec3 const& lambdas) {
    return
      "#version 330\n" +
      definitions +
      "const int TRANSMITTANCE_TEXTURE_WIDTH = "      + cs::utils::toString(mParams.mTransmittanceTextureWidth) + ";\n" +
      "const int TRANSMITTANCE_TEXTURE_HEIGHT = "     + cs::utils::toString(mParams.mTransmittanceTextureHeight) + ";\n" +
      "const int SCATTERING_TEXTURE_R_SIZE = "        + cs::utils::toString(mParams.mScatteringTextureRSize) + ";\n" +
      "const int SCATTERING_TEXTURE_MU_SIZE = "       + cs::utils::toString(mParams.mScatteringTextureMuSize) + ";\n" +
      "const int SCATTERING_TEXTURE_MU_S_SIZE = "     + cs::utils::toString(mParams.mScatteringTextureMuSSize) + ";\n" +
      "const int SCATTERING_TEXTURE_NU_SIZE = "       + cs::utils::toString(mParams.mScatteringTextureNuSize) + ";\n" +
      "const int IRRADIANCE_TEXTURE_WIDTH = "         + cs::utils::toString(mParams.mIrradianceTextureWidth) + ";\n" +
      "const int IRRADIANCE_TEXTURE_HEIGHT = "        + cs::utils::toString(mParams.mIrradianceTextureHeight) + ";\n" +
      "const int SAMPLE_COUNT_OPTICAL_DEPTH = "       + cs::utils::toString(mParams.mSampleCountOpticalDepth) + ";\n" +
      "const int SAMPLE_COUNT_SINGLE_SCATTERING = "   + cs::utils::toString(mParams.mSampleCountSingleScattering) + ";\n" +
      "const int SAMPLE_COUNT_SCATTERING_DENSITY = "  + cs::utils::toString(mParams.mSampleCountScatteringDensity) + ";\n" +
      "const int SAMPLE_COUNT_MULTI_SCATTERING = "    + cs::utils::toString(mParams.mSampleCountMultiScattering) + ";\n" +
      "const int SAMPLE_COUNT_INDIRECT_IRRADIANCE = " + cs::utils::toString(mParams.mSampleCountIndirectIrradiance) + ";\n" +
      "const vec3 SKY_SPECTRAL_RADIANCE_TO_LUMINANCE = vec3(" + cs::utils::toString(skyKR) + "," + cs::utils::toString(skyKG) + "," + cs::utils::toString(skyKB) + ");\n" +
      "const vec3 SUN_SPECTRAL_RADIANCE_TO_LUMINANCE = vec3(" + cs::utils::toString(sunKR) + "," + cs::utils::toString(sunKG) + "," + cs::utils::toString(sunKB) + ");\n" +
      "const vec3 SOLAR_IRRADIANCE = "                + extractVec3(WAVELENGTHS, SOLAR_IRRADIANCE, lambdas) + ";\n" +
      "const vec3 GROUND_ALBEDO = vec3("              + cs::utils::toString(mParams.mGroundAlbedo) + ");\n" +
      "const float SUN_ANGULAR_RADIUS = "             + cs::utils::toString(mParams.mSunAngularRadius) + ";\n" +
      "const float BOTTOM_RADIUS = "                  + cs::utils::toString(mParams.mBottomRadius) + ";\n" +
      "const float TOP_RADIUS = "                     + cs::utils::toString(mParams.mTopRadius) + ";\n" +
      "const float MU_S_MIN = "                       + cs::utils::toString(std::cos(mParams.mMaxSunZenithAngle))+ ";\n" +
      "const AtmosphereComponents ATMOSPHERE = AtmosphereComponents(\n" +
        printScatteringComponent(mParams.mMolecules, mParams.mWavelengths, 0.0, 0.0, lambdas) + ",\n" +
        printScatteringComponent(mParams.mAerosols, mParams.mWavelengths, 1.0, 0.5, lambdas) + ",\n" +
        printAbsorbingComponent(mParams.mOzone, mParams.mWavelengths, 1.0, lambdas) + ");\n" +
      functions;
    };
  // clang-format on

  // Allocate the precomputed textures, but don't precompute them yet.
  mTransmittanceTexture            = NewTexture2d(mParams.mTransmittanceTextureWidth,
      mParams.mTransmittanceTextureHeight, GL_RGBA32F, GL_RGBA, GL_FLOAT);
  mMultipleScatteringTexture       = NewTexture3d(mScatteringTextureWidth, mScatteringTextureHeight,
      mScatteringTextureDepth, GL_RGBA32F, GL_RGBA, GL_FLOAT);
  mSingleAerosolsScatteringTexture = NewTexture3d(mScatteringTextureWidth, mScatteringTextureHeight,
      mScatteringTextureDepth, GL_RGBA32F, GL_RGBA, GL_FLOAT);

  mIrradianceTexture = NewTexture2d(mParams.mIrradianceTextureWidth,
      mParams.mIrradianceTextureHeight, GL_RGBA32F, GL_RGBA, GL_FLOAT);

  // Create the density profile texture. It contains three rows of pixels, one for each constituent
  // of the atmosphere. The bottom-most density values are on the left, the top-most density values
  // are on the right.
  {
    size_t numDensities = mParams.mMolecules.mDensity.size();

    std::vector<float> densityData;
    size_t             numComponents = 3; // Rayleigh, Mie, Ozone
    densityData.reserve(numComponents * numDensities);

    densityData.insert(
        densityData.end(), mParams.mMolecules.mDensity.begin(), mParams.mMolecules.mDensity.end());
    densityData.insert(
        densityData.end(), mParams.mAerosols.mDensity.begin(), mParams.mAerosols.mDensity.end());
    densityData.insert(
        densityData.end(), mParams.mOzone.mDensity.begin(), mParams.mOzone.mDensity.end());

    mDensityTexture =
        NewTexture2d(numDensities, numComponents, GL_R32F, GL_RED, GL_FLOAT, densityData.data());
  }

  // Create and compile the shader providing our API.
  std::string shader = mGlslHeaderFactory({kLambdaR, kLambdaG, kLambdaB}) + kAtmosphereShader;
  const char* source = shader.c_str();
  mAtmosphereShader  = glCreateShader(GL_FRAGMENT_SHADER);
  glShaderSource(mAtmosphereShader, 1, &source, NULL);
  glCompileShader(mAtmosphereShader);

  // Create a full screen quad vertex array and vertex buffer objects.
  glGenVertexArrays(1, &mFullScreenQuadVAO);
  glBindVertexArray(mFullScreenQuadVAO);
  glGenBuffers(1, &mFullScreenQuadVBO);
  glBindBuffer(GL_ARRAY_BUFFER, mFullScreenQuadVBO);
  const GLfloat vertices[] = {
      -1.0,
      -1.0,
      +1.0,
      -1.0,
      -1.0,
      +1.0,
      +1.0,
      +1.0,
  };
  constexpr int kCoordsPerVertex = 2;
  glBufferData(GL_ARRAY_BUFFER, sizeof vertices, vertices, GL_STATIC_DRAW);
  constexpr GLuint kAttribIndex = 0;
  glVertexAttribPointer(kAttribIndex, kCoordsPerVertex, GL_FLOAT, false, 0, 0);
  glEnableVertexAttribArray(kAttribIndex);
  glBindVertexArray(0);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

Implementation::~Implementation() {
  glDeleteBuffers(1, &mFullScreenQuadVBO);
  glDeleteVertexArrays(1, &mFullScreenQuadVAO);
  glDeleteTextures(1, &mPhaseTexture);
  glDeleteTextures(1, &mTransmittanceTexture);
  glDeleteTextures(1, &mMultipleScatteringTexture);
  glDeleteTextures(1, &mSingleAerosolsScatteringTexture);
  glDeleteTextures(1, &mIrradianceTexture);
  glDeleteShader(mAtmosphereShader);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Functionality-wise, this is almost identical to the original implementation. There are some minor
// changes, for instance do we need to bind the density distribution texture for computing the
// transmittance. Also, we need to update the phase function texture after the last iteration of the
// pre-computation.

void Implementation::init(unsigned int numScatteringOrders) {
  // The precomputations require temporary textures, in particular to store the contribution of one
  // scattering order, which is needed to compute the next order of scattering (the final
  // precomputed textures store the sum of all the scattering orders). We allocate them here, and
  // destroy them at the end of this method.
  GLuint deltaIrradianceTexture          = NewTexture2d(mParams.mIrradianceTextureWidth,
      mParams.mIrradianceTextureHeight, GL_RGBA32F, GL_RGBA, GL_FLOAT);
  GLuint deltaMoleculesScatteringTexture = NewTexture3d(mScatteringTextureWidth,
      mScatteringTextureHeight, mScatteringTextureDepth, GL_RGBA32F, GL_RGBA, GL_FLOAT);
  GLuint deltaAerosolsScatteringTexture  = NewTexture3d(mScatteringTextureWidth,
      mScatteringTextureHeight, mScatteringTextureDepth, GL_RGBA32F, GL_RGBA, GL_FLOAT);
  GLuint deltaScatteringDensityTexture   = NewTexture3d(mScatteringTextureWidth,
      mScatteringTextureHeight, mScatteringTextureDepth, GL_RGBA32F, GL_RGBA, GL_FLOAT);

  // deltaMultipleScatteringTexture is only needed to compute scattering order 3 or more, while
  // deltaMoleculesScatteringTexture and deltaAerosolsScatteringTexture are only needed to
  // compute double scattering. Therefore, to save memory, we can store
  // deltaMoleculesScatteringTexture and deltaMultipleScatteringTexture in the same GPU
  // texture.
  GLuint deltaMultipleScatteringTexture = deltaMoleculesScatteringTexture;

  // The precomputations also require a temporary framebuffer object, created here (and destroyed at
  // the end of this method).
  GLuint fbo;
  glGenFramebuffers(1, &fbo);
  glBindFramebuffer(GL_FRAMEBUFFER, fbo);

  // The actual precomputations depend on whether we want to store precomputed irradiance or
  // illuminance values.
  if (mParams.mWavelengths.size() <= 3) {
    logger().info("Precomputing atmospheric scattering (1/1)...");
    glm::dvec3 lambdas{kLambdaR, kLambdaG, kLambdaB};
    glm::mat3  luminanceFromRadiance{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
    precompute(fbo, deltaIrradianceTexture, deltaMoleculesScatteringTexture,
        deltaAerosolsScatteringTexture, deltaScatteringDensityTexture,
        deltaMultipleScatteringTexture, lambdas, luminanceFromRadiance, false /* blend */,
        numScatteringOrders);
  } else {
    int numIterations = mParams.mWavelengths.size() / 3;
    for (int i = 0; i < numIterations; ++i) {
      logger().info("Precomputing atmospheric scattering ({}/{})...", i + 1, numIterations);

      glm::dvec3 lambdas{mParams.mWavelengths[i * 3 + 0], mParams.mWavelengths[i * 3 + 1],
          mParams.mWavelengths[i * 3 + 2]};

      auto coeff = [this](double lambda, int component) {
        // Note that we don't include MAX_LUMINOUS_EFFICACY here, to avoid artefacts due to too
        // large values when using half precision on GPU. We add this term back in
        // kAtmosphereShader, via SKY_SPECTRAL_RADIANCE_TO_LUMINANCE (see also the comments in the
        // Model constructor).
        double x = CieColorMatchingFunctionTableValue(lambda, 1);
        double y = CieColorMatchingFunctionTableValue(lambda, 2);
        double z = CieColorMatchingFunctionTableValue(lambda, 3);
        return static_cast<float>(
            (XYZ_TO_SRGB[component * 3] * x + XYZ_TO_SRGB[component * 3 + 1] * y +
                XYZ_TO_SRGB[component * 3 + 2] * z) *
            (mParams.mWavelengths[1] - mParams.mWavelengths[0]));
      };

      glm::mat3 luminanceFromRadiance{coeff(lambdas[0], 0), coeff(lambdas[1], 0),
          coeff(lambdas[2], 0), coeff(lambdas[0], 1), coeff(lambdas[1], 1), coeff(lambdas[2], 1),
          coeff(lambdas[0], 2), coeff(lambdas[1], 2), coeff(lambdas[2], 2)};

      precompute(fbo, deltaIrradianceTexture, deltaMoleculesScatteringTexture,
          deltaAerosolsScatteringTexture, deltaScatteringDensityTexture,
          deltaMultipleScatteringTexture, lambdas, luminanceFromRadiance, i > 0 /* blend */,
          numScatteringOrders);
    }

    // After the above iterations, the transmittance texture contains the transmittance for the 3
    // wavelengths used at the last iteration. But we want the transmittance at kLambdaR, kLambdaG,
    // kLambdaB instead, so we must recompute it here for these 3 wavelengths:
    std::string header = mGlslHeaderFactory({kLambdaR, kLambdaG, kLambdaB});
    Program     computeTransmittance(kVertexShader, header + kComputeTransmittanceShader);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, mTransmittanceTexture, 0);
    glDrawBuffer(GL_COLOR_ATTACHMENT0);
    glViewport(0, 0, mParams.mTransmittanceTextureWidth, mParams.mTransmittanceTextureHeight);
    glScissor(0, 0, mParams.mTransmittanceTextureWidth, mParams.mTransmittanceTextureHeight);
    computeTransmittance.Use();
    computeTransmittance.BindTexture2d("uDensityTexture", mDensityTexture, 0);
    DrawQuad({false}, mFullScreenQuadVAO);

    glFlush();

    // Also, the mPhaseTexture contains the phase functions for the last used wavelengths. We need
    // to update it with kLambdaR, kLambdaG, kLambdaB as well.
    updatePhaseFunctionTexture(
        {mParams.mMolecules, mParams.mAerosols}, {kLambdaR, kLambdaG, kLambdaB});
  }

  // Delete the temporary resources allocated at the begining of this method.
  glUseProgram(0);
  glBindFramebuffer(GL_FRAMEBUFFER, 0);
  glDeleteFramebuffers(1, &fbo);
  glDeleteTextures(1, &mDensityTexture);
  glDeleteTextures(1, &deltaScatteringDensityTexture);
  glDeleteTextures(1, &deltaAerosolsScatteringTexture);
  glDeleteTextures(1, &deltaMoleculesScatteringTexture);
  glDeleteTextures(1, &deltaIrradianceTexture);
  assert(glGetError() == 0);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

GLuint Implementation::shader() const {
  return mAtmosphereShader;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// This now binds one additional texture: The phase-function texture. The density-distribution
// texture is not required at run-time.

void Implementation::setProgramUniforms(GLuint program, GLuint phaseTextureUnit,
    GLuint transmittanceTextureUnit, GLuint multipleScatteringTextureUnit,
    GLuint irradianceTextureUnit, GLuint singleAerosolsScatteringTextureUnit) const {

  glActiveTexture(GL_TEXTURE0 + phaseTextureUnit);
  glBindTexture(GL_TEXTURE_2D, mPhaseTexture);
  glUniform1i(glGetUniformLocation(program, "uPhaseTexture"), phaseTextureUnit);

  glActiveTexture(GL_TEXTURE0 + transmittanceTextureUnit);
  glBindTexture(GL_TEXTURE_2D, mTransmittanceTexture);
  glUniform1i(glGetUniformLocation(program, "uTransmittanceTexture"), transmittanceTextureUnit);

  glActiveTexture(GL_TEXTURE0 + multipleScatteringTextureUnit);
  glBindTexture(GL_TEXTURE_3D, mMultipleScatteringTexture);
  glUniform1i(
      glGetUniformLocation(program, "uMultipleScatteringTexture"), multipleScatteringTextureUnit);

  glActiveTexture(GL_TEXTURE0 + irradianceTextureUnit);
  glBindTexture(GL_TEXTURE_2D, mIrradianceTexture);
  glUniform1i(glGetUniformLocation(program, "uIrradianceTexture"), irradianceTextureUnit);

  glActiveTexture(GL_TEXTURE0 + singleAerosolsScatteringTextureUnit);
  glBindTexture(GL_TEXTURE_3D, mSingleAerosolsScatteringTexture);
  glUniform1i(glGetUniformLocation(program, "uSingleAerosolsScatteringTexture"),
      singleAerosolsScatteringTextureUnit);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

/*
The actual pre-computation is also almost identical to the original implementation. The main
differences are that we now updated the phase-function texture before each iteration and have to
bind it together with the density-distribution texture in some of the steps.

You can have a look at the original implementation along with some explanations online at the bottom
of this page:
https://ebruneton.github.io/precomputed_atmospheric_scattering/atmosphere/model.cc.html#implementation

To help understanding the process, here is an outline of the data flow of the pre-computation. The
pre-computation is executed for batches of three wavelengths at a time.

1. Compute the transmittance (a vec3) of the atmosphere for every point in every direction and store
   it in mTransmittanceTexture. This incorporates extinction based on molecules, aerosols, and
   ozone particles.

2. Using mTransmittanceTexture, compute the direct irradiance from the Sun to every point in the
   atmosphere for the current set of wavelengths and store it in deltaIrradianceTexture. In this
   step, mIrradianceTexture is also initialized to zero if it's the first call to Precompute().

3. Using the mTransmittanceTexture, compute the single-aerosol and single-molecule scattering
   irradiance along the rays in the atmosphere. This is the molecules' and aerosols' density * solar
   irradiance * scattering coefficient. The term stored in the output textures is without the phase
   function. The irradiance for the current set of wavelengths is stored in
   deltaMoleculesScatteringTexture and deltaAerosolsScatteringTexture. It is also converted to
   illuminance and accumulated for all wavelengths in mMultipleScatteringTexture and
   mSingleAerosolsScatteringTexture.

At this point, mMultipleScatteringTexture and mSingleAerosolsScatteringTexture contain single
scattering illuminance without the phase function.

4. Iteratively compute higher orders of scattering. The following happens in a loop:

   4.1. Compute the scattering density, and store it in deltaScatteringDensityTexture.

   4.2. Compute the indirect irradiance, store it in deltaIrradianceTexture and accumulate it in
        mIrradianceTexture.

   4.3. Compute the multiple scattering, store it in deltaMultipleScatteringTexture, and
        accumulate it in mMultipleScatteringTexture.

At the end, mSingleAerosolsScatteringTexture contains the single-aerosol scattering illuminance
without the phase function and mMultipleScatteringTexture contains single-molecule scattering
without the phase function + multiple scattering with only the aerosols phase function applied. So
at render time, the data from mSingleAerosolsScatteringTexture needs to be multiplied with the
aerosols phase function and the data from mMultipleScatteringTexture needs to be multiplied with
the molecules phase function.
*/

void Implementation::precompute(GLuint fbo, GLuint deltaIrradianceTexture,
    GLuint deltaMoleculesScatteringTexture, GLuint deltaAerosolsScatteringTexture,
    GLuint deltaScatteringDensityTexture, GLuint deltaMultipleScatteringTexture,
    glm::dvec3 const& lambdas, glm::mat3 const& luminanceFromRadiance, bool blend,
    unsigned int numScatteringOrders) {

  // The precomputations require specific GLSL programs, for each precomputation step. We create and
  // compile them here (they are automatically destroyed when this method returns, via the Program
  // destructor).
  std::string header = mGlslHeaderFactory(lambdas);

  Program computeTransmittance(kVertexShader, header + kComputeTransmittanceShader);
  Program computeDirectIrradiance(kVertexShader, header + kComputeDirectIrradianceShader);
  Program computeSingleScattering(
      kVertexShader, kGeometryShader, header + kComputeSingleScatteringShader);
  Program computeScatteringDensity(
      kVertexShader, kGeometryShader, header + kComputeScatteringDensityShader);
  Program computeIndirectIrradiance(kVertexShader, header + kComputeIndirectIrradianceShader);
  Program computeMultipleScattering(
      kVertexShader, kGeometryShader, header + kComputeMultipleScatteringShader);

  const GLuint kDrawBuffers[4] = {
      GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1, GL_COLOR_ATTACHMENT2, GL_COLOR_ATTACHMENT3};
  glBlendEquationSeparate(GL_FUNC_ADD, GL_FUNC_ADD);
  glBlendFuncSeparate(GL_ONE, GL_ONE, GL_ONE, GL_ONE);

  // -----------------------------------------------------------------------------------------------

  updatePhaseFunctionTexture({mParams.mMolecules, mParams.mAerosols}, lambdas);

  // -----------------------------------------------------------------------------------------------

  // 1. Compute the transmittance, and store it in mTransmittanceTexture.
  glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, mTransmittanceTexture, 0);
  glDrawBuffer(GL_COLOR_ATTACHMENT0);
  glViewport(0, 0, mParams.mTransmittanceTextureWidth, mParams.mTransmittanceTextureHeight);
  glScissor(0, 0, mParams.mTransmittanceTextureWidth, mParams.mTransmittanceTextureHeight);
  computeTransmittance.Use();
  computeTransmittance.BindTexture2d("uDensityTexture", mDensityTexture, 0);
  DrawQuad({false}, mFullScreenQuadVAO);

  // -----------------------------------------------------------------------------------------------

  // 2. Compute the direct irradiance, store it in deltaIrradianceTexture and, depending on
  // 'blend', either initialize mIrradianceTexture with zeros or leave it unchanged (we don't want
  // the direct irradiance in mIrradianceTexture, but only the irradiance from the sky).
  glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, deltaIrradianceTexture, 0);
  glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, mIrradianceTexture, 0);
  glDrawBuffers(2, kDrawBuffers);
  glViewport(0, 0, mParams.mIrradianceTextureWidth, mParams.mIrradianceTextureHeight);
  glScissor(0, 0, mParams.mIrradianceTextureWidth, mParams.mIrradianceTextureHeight);
  computeDirectIrradiance.Use();
  computeDirectIrradiance.BindTexture2d("uTransmittanceTexture", mTransmittanceTexture, 0);
  DrawQuad({false, blend}, mFullScreenQuadVAO);

  // -----------------------------------------------------------------------------------------------

  // 3. Compute the molecules and aerosols single scattering for the current wavelengths, store them
  // in deltaMoleculesScatteringTexture and deltaAerosolsScatteringTexture, and accumulate the
  // resulting luminance via additive blending in mMultipleScatteringTexture and
  // mSingleAerosolsScatteringTexture. The molecules scattering is stored together with the
  // multiple scattering contributions in mMultipleScatteringTexture.
  glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, deltaMoleculesScatteringTexture, 0);
  glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, deltaAerosolsScatteringTexture, 0);
  glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT2, mMultipleScatteringTexture, 0);
  glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT3, mSingleAerosolsScatteringTexture, 0);
  glDrawBuffers(4, kDrawBuffers);

  glViewport(0, 0, mScatteringTextureWidth, mScatteringTextureHeight);
  glScissor(0, 0, mScatteringTextureWidth, mScatteringTextureHeight);
  computeSingleScattering.Use();
  computeSingleScattering.BindMat3("uLuminanceFromRadiance", luminanceFromRadiance);
  computeSingleScattering.BindTexture2d("uTransmittanceTexture", mTransmittanceTexture, 0);
  computeSingleScattering.BindTexture2d("uDensityTexture", mDensityTexture, 1);
  for (int32_t layer = 0; layer < mScatteringTextureDepth; ++layer) {
    computeSingleScattering.BindInt("uLayer", layer);
    DrawQuad({false, false, blend, blend}, mFullScreenQuadVAO);
  }

  // -----------------------------------------------------------------------------------------------

  // 4. Compute the 2nd, 3rd and 4th order of scattering, in sequence.
  for (unsigned int scatteringOrder = 2; scatteringOrder <= numScatteringOrders;
       ++scatteringOrder) {
    // 4.1. Compute the scattering density, and store it in deltaScatteringDensityTexture.
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, deltaScatteringDensityTexture, 0);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, 0, 0);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT2, 0, 0);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT3, 0, 0);
    glDrawBuffer(GL_COLOR_ATTACHMENT0);
    glViewport(0, 0, mScatteringTextureWidth, mScatteringTextureHeight);
    glScissor(0, 0, mScatteringTextureWidth, mScatteringTextureHeight);
    computeScatteringDensity.Use();
    computeScatteringDensity.BindTexture2d("uPhaseTexture", mPhaseTexture, 0);
    computeScatteringDensity.BindTexture2d("uTransmittanceTexture", mTransmittanceTexture, 1);
    computeScatteringDensity.BindTexture2d("uDensityTexture", mDensityTexture, 2);
    computeScatteringDensity.BindTexture3d(
        "uSingleMoleculesScatteringTexture", deltaMoleculesScatteringTexture, 3);
    computeScatteringDensity.BindTexture3d(
        "uSingleAerosolsScatteringTexture", deltaAerosolsScatteringTexture, 4);
    computeScatteringDensity.BindTexture3d(
        "uMultipleScatteringTexture", deltaMultipleScatteringTexture, 5);
    computeScatteringDensity.BindTexture2d("uIrradianceTexture", deltaIrradianceTexture, 6);
    computeScatteringDensity.BindInt("uScatteringOrder", scatteringOrder);
    for (int32_t layer = 0; layer < mScatteringTextureDepth; ++layer) {
      computeScatteringDensity.BindInt("uLayer", layer);
      DrawQuad({false}, mFullScreenQuadVAO);
    }

    // 4.2. Compute the indirect irradiance, store it in deltaIrradianceTexture and accumulate it
    // in mIrradianceTexture.
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, deltaIrradianceTexture, 0);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, mIrradianceTexture, 0);
    glDrawBuffers(2, kDrawBuffers);
    glViewport(0, 0, mParams.mIrradianceTextureWidth, mParams.mIrradianceTextureHeight);
    glScissor(0, 0, mParams.mIrradianceTextureWidth, mParams.mIrradianceTextureHeight);
    computeIndirectIrradiance.Use();
    computeIndirectIrradiance.BindMat3("uLuminanceFromRadiance", luminanceFromRadiance);
    computeIndirectIrradiance.BindTexture2d("uPhaseTexture", mPhaseTexture, 0);
    computeIndirectIrradiance.BindTexture3d(
        "uSingleMoleculesScatteringTexture", deltaMoleculesScatteringTexture, 1);
    computeIndirectIrradiance.BindTexture3d(
        "uSingleAerosolsScatteringTexture", deltaAerosolsScatteringTexture, 2);
    computeIndirectIrradiance.BindTexture3d(
        "uMultipleScatteringTexture", deltaMultipleScatteringTexture, 3);
    computeIndirectIrradiance.BindInt("uScatteringOrder", scatteringOrder - 1);
    DrawQuad({false, true}, mFullScreenQuadVAO);

    // 4.3. Compute the multiple scattering, store it in deltaMultipleScatteringTexture, and
    // accumulate it in mMultipleScatteringTexture.
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, deltaMultipleScatteringTexture, 0);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, mMultipleScatteringTexture, 0);
    glDrawBuffers(2, kDrawBuffers);
    glViewport(0, 0, mScatteringTextureWidth, mScatteringTextureHeight);
    glScissor(0, 0, mScatteringTextureWidth, mScatteringTextureHeight);
    computeMultipleScattering.Use();
    computeMultipleScattering.BindMat3("uLuminanceFromRadiance", luminanceFromRadiance);
    computeMultipleScattering.BindTexture2d("uPhaseTexture", mPhaseTexture, 0);
    computeMultipleScattering.BindTexture2d("uTransmittanceTexture", mTransmittanceTexture, 1);
    computeMultipleScattering.BindTexture3d(
        "uScatteringDensityTexture", deltaScatteringDensityTexture, 2);
    for (int32_t layer = 0; layer < mScatteringTextureDepth; ++layer) {
      computeMultipleScattering.BindInt("uLayer", layer);
      DrawQuad({false, true}, mFullScreenQuadVAO);
    }
  }

  glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, 0, 0);
  glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT2, 0, 0);
  glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT3, 0, 0);

  glFlush();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// This final method is used to update the phase function texture according to the currently
// pre-computed wavelengths. It extracts the corresponding intensity values and stores them in a 2D
// texture. Each row of pixels corresponds to one scattering component. Forward-scattering is on the
// left, back-scattering is on the right.

void Implementation::updatePhaseFunctionTexture(
    std::vector<Params::ScatteringComponent> const& scatteringComponents,
    glm::dvec3 const&                               lambdas) {

  if (mPhaseTexture != 0) {
    glDeleteTextures(1, &mPhaseTexture);
  }

  size_t numAngles = scatteringComponents.front().mPhase.size();

  std::vector<float> data;
  data.reserve(4 * scatteringComponents.size() * numAngles);

  for (size_t i(0); i < scatteringComponents.size(); ++i) {
    for (auto const& spectrum : scatteringComponents[i].mPhase) {
      data.push_back(Interpolate(mParams.mWavelengths, spectrum, lambdas[0]));
      data.push_back(Interpolate(mParams.mWavelengths, spectrum, lambdas[1]));
      data.push_back(Interpolate(mParams.mWavelengths, spectrum, lambdas[2]));
      data.push_back(0.f);
    }
  }

  mPhaseTexture = NewTexture2d(
      numAngles, scatteringComponents.size(), GL_RGBA32F, GL_RGBA, GL_FLOAT, data.data());
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::atmospheres::models::bruneton::internal
