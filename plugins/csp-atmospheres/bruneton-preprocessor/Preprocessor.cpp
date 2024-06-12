////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-FileCopyrightText: 2017 Eric Bruneton
// SPDX-License-Identifier: BSD-3-Clause

#include "Preprocessor.hpp"

#include "../../../../src/cs-utils/filesystem.hpp"
#include "../../../../src/cs-utils/utils.hpp"

#include <cassert>
#include <cmath>
#include <glm/gtc/type_ptr.hpp>
#include <iostream>
#include <memory>
#include <tiffio.h>

// This file is based in large parts on the original implementation by Eric Bruneton:
// https://github.com/ebruneton/precomputed_atmospheric_scattering/blob/master/atmosphere/model.cc

// While implementing the atmospheric model into CosmoScout VR, we have refactored some parts of the
// code, however this is mostly related to how variables are named and how input parameters are
// passed to the model.
// Architecture-wise, the main difference is that the preprocessing is now done offline, so all code
// which is only required during rendering has been refactored out. Functionality-wise, the only
// fundamental change is that the phase functions for aerosols and molecules as well as their
// density distributions are now loaded from CSV files and then later sampled from textures. We also
// store photometric values instead of radiometric values in the final textures.

// Below, we will indicate for each group of function whether something has been changed and a link
// to the original explanations of the methods by Eric Bruneton.

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

// Shader Definitions ------------------------------------------------------------------------------

// Below, the source code for several shaders is defined.

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
  layout(location = 1) out vec3 oIlluminance;

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
    oIlluminance = uLuminanceFromRadiance * oDeltaIrradiance;
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
// for the availability of RGB textures anymore.

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
  float solarR  = Interpolate(WAVELENGTHS, SOLAR_IRRADIANCE, Preprocessor::kLambdaR);
  float solarG  = Interpolate(WAVELENGTHS, SOLAR_IRRADIANCE, Preprocessor::kLambdaG);
  float solarB  = Interpolate(WAVELENGTHS, SOLAR_IRRADIANCE, Preprocessor::kLambdaB);
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
    *kR += r_bar * irradiance / solarR * pow(lambda / Preprocessor::kLambdaR, lambdaPower);
    *kG += g_bar * irradiance / solarG * pow(lambda / Preprocessor::kLambdaG, lambdaPower);
    *kB += b_bar * irradiance / solarB * pow(lambda / Preprocessor::kLambdaB, lambdaPower);
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

// This creates an GLSL snippet corresponding to the given scattering component.
std::string printScatteringComponent(Params::ScatteringComponent const& component,
    std::vector<float> const& wavelengths, float phaseTextureV, float densityTextureV,
    glm::vec3 const& lambdas) {

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
    std::vector<float> const& wavelengths, float densityTextureV, glm::vec3 const& lambdas) {

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

// Preprocessor Implementation ---------------------------------------------------------------------

// The code below roughly follows the original implementation by Eric Bruneton.

// The original explanation of the methods still applies in most parts and is available online:
// https://ebruneton.github.io/precomputed_atmospheric_scattering/atmosphere/model.cc.html#implementation

// The main differences in the constructor are that we pass significantly more constants to the
// shader code as all the sampling counts are configurable now. Also, we create the new density
// texture which contains the density profiles for all atmosphere constituents.
Preprocessor::Preprocessor(Params params)
    : mParams(std::move(params))
    , mScatteringTextureWidth(
          mParams.mScatteringTextureNuSize.get() * mParams.mScatteringTextureMuSSize.get())
    , mScatteringTextureHeight(mParams.mScatteringTextureMuSize.get())
    , mScatteringTextureDepth(mParams.mScatteringTextureRSize.get()) {

  std::cout << "Preprocessing atmosphere..." << std::endl;

  // Compute angular radius of the sun.
  float sunRadius             = 696340000; // meters
  mMetadata.mSunAngularRadius = std::asin(sunRadius / mParams.mSunDistance);

  // Compute the values for the SUN_RADIANCE_TO_LUMINANCE constant.
  float sunAngularRadiusAtEarth = 0.0046547; // radians
  float attenuation =
      std::pow(mMetadata.mSunAngularRadius, 2.F) / std::pow(sunAngularRadiusAtEarth, 2.F);
  float sunKR, sunKG, sunKB;
  ComputeSpectralRadianceToLuminanceFactors(0 /* lambdaPower */, &sunKR, &sunKG, &sunKB);
  sunKR *= Interpolate(WAVELENGTHS, SOLAR_IRRADIANCE, kLambdaR) * attenuation;
  sunKG *= Interpolate(WAVELENGTHS, SOLAR_IRRADIANCE, kLambdaG) * attenuation;
  sunKB *= Interpolate(WAVELENGTHS, SOLAR_IRRADIANCE, kLambdaB) * attenuation;

  mMetadata.mSunIlluminance          = glm::vec3(sunKR, sunKG, sunKB);
  mMetadata.mScatteringTextureNuSize = mParams.mScatteringTextureNuSize.get();
  mMetadata.mMaxSunZenithAngle       = mParams.mMaxSunZenithAngle.get();

  // A lambda that creates a GLSL header containing our atmosphere computation functions,
  // specialized for the given atmosphere parameters and for the 3 wavelengths in 'lambdas'.
  auto definitions =
      cs::utils::filesystem::loadToString("plugins/csp-atmospheres/bruneton-preprocessor/shaders/"
                                          "csp-atmosphere-preprocessing-definitions.glsl");
  auto functions =
      cs::utils::filesystem::loadToString("plugins/csp-atmospheres/bruneton-preprocessor/shaders/"
                                          "csp-atmosphere-preprocessing-functions.glsl");
  auto common = cs::utils::filesystem::loadToString(
      "plugins/csp-atmospheres/shaders/atmosphere-models/bruneton/common.glsl");

  // clang-format off
  mGlslHeaderFactory = [=](glm::vec3 const& lambdas) {
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
      "const vec3 SOLAR_IRRADIANCE = "                + extractVec3(WAVELENGTHS, SOLAR_IRRADIANCE, lambdas) + ";\n" +
      "const vec3 GROUND_ALBEDO = vec3("              + cs::utils::toString(mParams.mGroundAlbedo) + ");\n" +
      "const float SUN_ANGULAR_RADIUS = "             + cs::utils::toString(mMetadata.mSunAngularRadius) + ";\n" +
      "const float BOTTOM_RADIUS = "                  + cs::utils::toString(mParams.mMinAltitude) + ";\n" +
      "const float TOP_RADIUS = "                     + cs::utils::toString(mParams.mMaxAltitude) + ";\n" +
      "const float MU_S_MIN = "                       + cs::utils::toString(std::cos(mParams.mMaxSunZenithAngle.get()))+ ";\n" +
      "const AtmosphereComponents ATMOSPHERE = AtmosphereComponents(\n" +
        printScatteringComponent(mParams.mMolecules, mParams.mWavelengths, 0.0, 0.0, lambdas) + ",\n" +
        printScatteringComponent(mParams.mAerosols, mParams.mWavelengths, 1.0, 0.5, lambdas) + ",\n" +
        printAbsorbingComponent(mParams.mOzone.value(), mParams.mWavelengths, 1.0, lambdas) + ");\n" +
      common + "\n" +
      functions;
    };
  // clang-format on

  // Allocate the precomputed textures, but don't precompute them yet.
  mTransmittanceTexture = NewTexture2d(mParams.mTransmittanceTextureWidth.get(),
      mParams.mTransmittanceTextureHeight.get(), GL_RGB32F, GL_RGB, GL_FLOAT);

  mMultipleScatteringTexture = NewTexture3d(mScatteringTextureWidth, mScatteringTextureHeight,
      mScatteringTextureDepth, GL_RGB32F, GL_RGB, GL_FLOAT);

  mSingleAerosolsScatteringTexture = NewTexture3d(mScatteringTextureWidth, mScatteringTextureHeight,
      mScatteringTextureDepth, GL_RGB32F, GL_RGB, GL_FLOAT);

  mIrradianceTexture = NewTexture2d(mParams.mIrradianceTextureWidth.get(),
      mParams.mIrradianceTextureHeight.get(), GL_RGB32F, GL_RGB, GL_FLOAT);

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
        densityData.end(), mParams.mOzone->mDensity.begin(), mParams.mOzone->mDensity.end());
    mDensityTexture = NewTexture2d(static_cast<int>(numDensities), static_cast<int>(numComponents),
        GL_R32F, GL_RED, GL_FLOAT, densityData.data());
  }

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

Preprocessor::~Preprocessor() {
  glDeleteBuffers(1, &mFullScreenQuadVBO);
  glDeleteVertexArrays(1, &mFullScreenQuadVAO);
  glDeleteTextures(1, &mPhaseTexture);
  glDeleteTextures(1, &mTransmittanceTexture);
  glDeleteTextures(1, &mMultipleScatteringTexture);
  glDeleteTextures(1, &mSingleAerosolsScatteringTexture);
  glDeleteTextures(1, &mIrradianceTexture);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Functionality-wise, this is almost identical to the original implementation. There are some minor
// changes, for instance do we need to bind the density distribution texture for computing the
// transmittance. Also, we need to update the phase function texture after the last iteration of the
// precomputation.

void Preprocessor::run(unsigned int numScatteringOrders) {
  // The precomputations require temporary textures, in particular to store the contribution of one
  // scattering order, which is needed to compute the next order of scattering (the final
  // precomputed textures store the sum of all the scattering orders). We allocate them here, and
  // destroy them at the end of this method.
  GLuint deltaIrradianceTexture = NewTexture2d(mParams.mIrradianceTextureWidth.get(),
      mParams.mIrradianceTextureHeight.get(), GL_RGB32F, GL_RGB, GL_FLOAT);

  GLuint deltaMoleculesScatteringTexture = NewTexture3d(mScatteringTextureWidth,
      mScatteringTextureHeight, mScatteringTextureDepth, GL_RGB32F, GL_RGB, GL_FLOAT);

  GLuint deltaAerosolsScatteringTexture = NewTexture3d(mScatteringTextureWidth,
      mScatteringTextureHeight, mScatteringTextureDepth, GL_RGB32F, GL_RGB, GL_FLOAT);

  GLuint deltaScatteringDensityTexture = NewTexture3d(mScatteringTextureWidth,
      mScatteringTextureHeight, mScatteringTextureDepth, GL_RGB32F, GL_RGB, GL_FLOAT);

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
    std::cout << "Precomputing atmospheric scattering (1/1)..." << std::endl;
    glm::vec3 lambdas{kLambdaR, kLambdaG, kLambdaB};

    float skyKR, skyKG, skyKB;
    ComputeSpectralRadianceToLuminanceFactors(-3 /* lambdaPower */, &skyKR, &skyKG, &skyKB);

    glm::mat3 luminanceFromRadiance{skyKR, 0.0, 0.0, 0.0, skyKG, 0.0, 0.0, 0.0, skyKB};
    precompute(fbo, deltaIrradianceTexture, deltaMoleculesScatteringTexture,
        deltaAerosolsScatteringTexture, deltaScatteringDensityTexture,
        deltaMultipleScatteringTexture, lambdas, luminanceFromRadiance, false /* blend */,
        numScatteringOrders);
  } else {
    int numIterations = static_cast<int>(mParams.mWavelengths.size()) / 3;
    for (int i = 0; i < numIterations; ++i) {
      std::cout << "Precomputing atmospheric scattering (" << i + 1 << "/" << numIterations
                << ")..." << std::endl;

      glm::vec3 lambdas{mParams.mWavelengths[i * 3 + 0], mParams.mWavelengths[i * 3 + 1],
          mParams.mWavelengths[i * 3 + 2]};

      auto coeff = [this](float lambda, int component) {
        // Note that we don't include MAX_LUMINOUS_EFFICACY here, to avoid artefacts due to too
        // large values when using half precision on GPU. We add this term back in
        // kAtmosphereShader, via SKY_SPECTRAL_RADIANCE_TO_LUMINANCE (see also the comments in the
        // Model constructor).
        float x = CieColorMatchingFunctionTableValue(lambda, 1);
        float y = CieColorMatchingFunctionTableValue(lambda, 2);
        float z = CieColorMatchingFunctionTableValue(lambda, 3);
        return static_cast<float>(
                   (XYZ_TO_SRGB[component * 3] * x + XYZ_TO_SRGB[component * 3 + 1] * y +
                       XYZ_TO_SRGB[component * 3 + 2] * z) *
                   (mParams.mWavelengths[1] - mParams.mWavelengths[0])) *
               MAX_LUMINOUS_EFFICACY;
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
    glViewport(
        0, 0, mParams.mTransmittanceTextureWidth.get(), mParams.mTransmittanceTextureHeight.get());
    glScissor(
        0, 0, mParams.mTransmittanceTextureWidth.get(), mParams.mTransmittanceTextureHeight.get());
    computeTransmittance.Use();
    computeTransmittance.BindTexture2d("uDensityTexture", mDensityTexture, 0);
    DrawQuad({false}, mFullScreenQuadVAO);

    glFlush();

    // Also, the mPhaseTexture contains the phase functions for the last used wavelengths. We need
    // to update it with kLambdaR, kLambdaG, kLambdaB as well.
    updatePhaseFunctionTexture(
        {mParams.mMolecules, mParams.mAerosols}, {kLambdaR, kLambdaG, kLambdaB});
  }

  // Delete the temporary resources allocated at the beginning of this method.
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

// This method did not exist in the original implementation. It saves the precomputed textures as
// tiff files to disk. The metadata is saved as a json file.

void Preprocessor::save(std::string const& directory) {
  std::cout << "Saving precomputed atmosphere to disk..." << std::endl;

  // Save the precomputed textures to disk. We need to store mMultipleScatteringTexture,
  // mSingleAerosolsScatteringTexture, mPhaseTexture, mTransmittanceTexture, and mIrradianceTexture.

  auto write2D = [](std::string const& path, GLuint texture, int width, int height) {
    std::vector<float> data(width * height * 3);
    glBindTexture(GL_TEXTURE_2D, texture);
    glGetTexImage(GL_TEXTURE_2D, 0, GL_RGB, GL_FLOAT, data.data());
    glBindTexture(GL_TEXTURE_2D, 0);

    auto* tiff = TIFFOpen(path.c_str(), "w");
    TIFFSetField(tiff, TIFFTAG_IMAGEWIDTH, width);
    TIFFSetField(tiff, TIFFTAG_IMAGELENGTH, height);
    TIFFSetField(tiff, TIFFTAG_SAMPLESPERPIXEL, 3);
    TIFFSetField(tiff, TIFFTAG_BITSPERSAMPLE, 32);
    TIFFSetField(tiff, TIFFTAG_SAMPLEFORMAT, SAMPLEFORMAT_IEEEFP);
    TIFFSetField(tiff, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_RGB);
    TIFFSetField(tiff, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
    TIFFSetField(tiff, TIFFTAG_COMPRESSION, COMPRESSION_DEFLATE);
    TIFFSetField(tiff, TIFFTAG_ROWSPERSTRIP, 1);
    for (int y = 0; y < height; ++y) {
      TIFFWriteScanline(tiff, data.data() + y * width * 3, y);
    }
    TIFFClose(tiff);
  };

  auto write3D = [](std::string const& path, GLuint texture, int width, int height, int depth) {
    std::vector<float> data(width * height * depth * 3);
    glBindTexture(GL_TEXTURE_3D, texture);
    glGetTexImage(GL_TEXTURE_3D, 0, GL_RGB, GL_FLOAT, data.data());
    glBindTexture(GL_TEXTURE_3D, 0);

    auto* tiff = TIFFOpen(path.c_str(), "w");

    for (int z = 0; z < depth; ++z) {
      TIFFSetField(tiff, TIFFTAG_PAGENUMBER, z, z);
      TIFFSetField(tiff, TIFFTAG_SUBFILETYPE, FILETYPE_PAGE);
      TIFFSetField(tiff, TIFFTAG_IMAGEWIDTH, width);
      TIFFSetField(tiff, TIFFTAG_IMAGELENGTH, height);
      TIFFSetField(tiff, TIFFTAG_SAMPLESPERPIXEL, 3);
      TIFFSetField(tiff, TIFFTAG_BITSPERSAMPLE, 32);
      TIFFSetField(tiff, TIFFTAG_SAMPLEFORMAT, SAMPLEFORMAT_IEEEFP);
      TIFFSetField(tiff, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_RGB);
      TIFFSetField(tiff, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
      TIFFSetField(tiff, TIFFTAG_COMPRESSION, COMPRESSION_DEFLATE);
      TIFFSetField(tiff, TIFFTAG_ROWSPERSTRIP, 1);
      for (int y = 0; y < height; ++y) {
        TIFFWriteScanline(tiff, data.data() + z * width * height * 3 + y * width * 3, y);
      }
      TIFFWriteDirectory(tiff);
    }
    TIFFClose(tiff);
  };

  size_t numAngles = mParams.mMolecules.mPhase.size();
  write2D(directory + "/phase.tif", mPhaseTexture, numAngles, 2);
  write2D(directory + "/transmittance.tif", mTransmittanceTexture,
      mParams.mTransmittanceTextureWidth.get(), mParams.mTransmittanceTextureHeight.get());
  write2D(directory + "/indirect_illuminance.tif", mIrradianceTexture,
      mParams.mIrradianceTextureWidth.get(), mParams.mIrradianceTextureHeight.get());
  write3D(directory + "/multiple_scattering.tif", mMultipleScatteringTexture,
      mScatteringTextureWidth, mScatteringTextureHeight, mScatteringTextureDepth);
  write3D(directory + "/single_aerosols_scattering.tif", mSingleAerosolsScatteringTexture,
      mScatteringTextureWidth, mScatteringTextureHeight, mScatteringTextureDepth);

  std::ofstream  out(directory + "/metadata.json");
  nlohmann::json data = mMetadata;
  out << std::setw(2) << data;

  std::cout << "Precomputed atmosphere saved to disk." << std::endl;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

/*
The actual precomputation is also almost identical to the original implementation. The main
differences are that we now updated the phase-function texture before each iteration and have to
bind it together with the density-distribution texture in some of the steps.

You can have a look at the original implementation along with some explanations online at the bottom
of this page:
https://ebruneton.github.io/precomputed_atmospheric_scattering/atmosphere/model.cc.html#implementation

To help understanding the process, here is an outline of the data flow of the precomputation. The
precomputation is executed for batches of three wavelengths at a time.

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

void Preprocessor::precompute(GLuint fbo, GLuint deltaIrradianceTexture,
    GLuint deltaMoleculesScatteringTexture, GLuint deltaAerosolsScatteringTexture,
    GLuint deltaScatteringDensityTexture, GLuint deltaMultipleScatteringTexture,
    glm::vec3 const& lambdas, glm::mat3 const& luminanceFromRadiance, bool blend,
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
  glViewport(
      0, 0, mParams.mTransmittanceTextureWidth.get(), mParams.mTransmittanceTextureHeight.get());
  glScissor(
      0, 0, mParams.mTransmittanceTextureWidth.get(), mParams.mTransmittanceTextureHeight.get());
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
  glViewport(0, 0, mParams.mIrradianceTextureWidth.get(), mParams.mIrradianceTextureHeight.get());
  glScissor(0, 0, mParams.mIrradianceTextureWidth.get(), mParams.mIrradianceTextureHeight.get());
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
    glViewport(0, 0, mParams.mIrradianceTextureWidth.get(), mParams.mIrradianceTextureHeight.get());
    glScissor(0, 0, mParams.mIrradianceTextureWidth.get(), mParams.mIrradianceTextureHeight.get());
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
// precomputed wavelengths. It extracts the corresponding intensity values and stores them in a 2D
// texture. Each row of pixels corresponds to one scattering component. Forward-scattering is on the
// left, back-scattering is on the right.

void Preprocessor::updatePhaseFunctionTexture(
    std::vector<Params::ScatteringComponent> const& scatteringComponents,
    glm::vec3 const&                                lambdas) {

  if (mPhaseTexture != 0) {
    glDeleteTextures(1, &mPhaseTexture);
  }

  size_t numAngles = scatteringComponents.front().mPhase.size();

  std::vector<float> data;
  data.reserve(3 * scatteringComponents.size() * numAngles);

  for (size_t i(0); i < scatteringComponents.size(); ++i) {
    for (auto const& spectrum : scatteringComponents[i].mPhase) {
      data.push_back(static_cast<float>(Interpolate(mParams.mWavelengths, spectrum, lambdas[0])));
      data.push_back(static_cast<float>(Interpolate(mParams.mWavelengths, spectrum, lambdas[1])));
      data.push_back(static_cast<float>(Interpolate(mParams.mWavelengths, spectrum, lambdas[2])));
    }
  }

  mPhaseTexture = NewTexture2d(static_cast<int>(numAngles),
      static_cast<int>(scatteringComponents.size()), GL_RGB32F, GL_RGB, GL_FLOAT, data.data());
}

////////////////////////////////////////////////////////////////////////////////////////////////////
