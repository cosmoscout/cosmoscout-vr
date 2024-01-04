////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-FileCopyrightText: 2017 Eric Bruneton
// SPDX-License-Identifier: BSD-3-Clause

// This file has been directly copied from here:
// https://github.com/ebruneton/precomputed_atmospheric_scattering/blob/master/atmosphere/model.cc
// The documentation below can also be read online at:
// https://ebruneton.github.io/precomputed_atmospheric_scattering/atmosphere/model.cc.html
// Changes to this file are mostly related to formatting. The only other change with respect to the
// original code is the removal of the "normal" parameter from the GetSunAndSkyIrradiance() and
// GetSunAndSkyIlluminance() methods. In the original implementation, these methods used to
// premultiply the irradiance with the dot product between light direction and surface normal. As
// this factor is already included in the BRDFs used in CosmoCout VR, we have removed this. Also,
// the GLSL files are now loaded via cs::utils::filesystem::loadToString().
// Also, the shadow_length parameter has been removed from the public API as this is currently
// not supported by CosmoScout VR.

/*<h2>atmosphere/model.cc</h2>

<p>This file implements the <a href="model.h.html">API of our atmosphere
model</a>. Its main role is to precompute the transmittance, scattering and
irradiance textures. The GLSL functions to precompute them are provided in
<a href="functions.glsl.html">functions.glsl</a>, but they are not sufficient.
They must be used in fully functional shaders and programs, and these programs
must be called in the correct order, with the correct input and output textures
(via framebuffer objects), to precompute each scattering order in sequence, as
described in Algorithm 4.1 of
<a href="https://hal.inria.fr/inria-00288758/en">our paper</a>. This is the role
of the following C++ code.
*/

#include "Model.hpp"

#include "../../../logger.hpp"
#include "CSVLoader.hpp"
#include "constants.hpp"

#include "../../../../src/cs-utils/filesystem.hpp"
#include "../../../../src/cs-utils/utils.hpp"

#include <cassert>
#include <cmath>
#include <iostream>
#include <memory>

// From the demo application by Eric Bruneton. The original source code can be found here:
// https://github.com/ebruneton/precomputed_atmospheric_scattering/blob/master/atmosphere/demo/demo.cc
// Values from "Reference Solar Spectral Irradiance: ASTM G-173", ETR column
// (see http://rredc.nrel.gov/solar/spectra/am1.5/ASTMG173/ASTMG173.html),
// summed and averaged in each bin (e.g. the value for 360nm is the average
// of the ASTM G-173 values for all wavelengths between 360 and 370nm).
// Values in W.m^-2.
const std::vector<double> WAVELENGTHS = {360, 370, 380, 390, 400, 410, 420, 430, 440, 450, 460, 470,
    480, 490, 500, 510, 520, 530, 540, 550, 560, 570, 580, 590, 600, 610, 620, 630, 640, 650, 660,
    670, 680, 690, 700, 710, 720, 730, 740, 750, 760, 770, 780, 790, 800, 810, 820, 830};
const std::vector<double> SOLAR_IRRADIANCE = {1.11776, 1.14259, 1.01249, 1.14716, 1.72765, 1.73054,
    1.6887, 1.61253, 1.91198, 2.03474, 2.02042, 2.02212, 1.93377, 1.95809, 1.91686, 1.8298, 1.8685,
    1.8931, 1.85149, 1.8504, 1.8341, 1.8345, 1.8147, 1.78158, 1.7533, 1.6965, 1.68194, 1.64654,
    1.6048, 1.52143, 1.55622, 1.5113, 1.474, 1.4482, 1.41018, 1.36775, 1.34188, 1.31429, 1.28303,
    1.26758, 1.2367, 1.2082, 1.18737, 1.14683, 1.12362, 1.1058, 1.07124, 1.04992};

/*
<p>The rest of this file is organized in 3 parts:
<ul>
<li>the <a href="#shaders">first part</a> defines the shaders used to precompute
the atmospheric textures,</li>
<li>the <a href="#utilities">second part</a> provides utility classes and
functions used to compile shaders, create textures, draw quads, etc,</li>
<li>the <a href="#implementation">third part</a> provides the actual
implementation of the <code>Model</code> class, using the above tools.</li>
</ul>

<h3 id="shaders">Shader definitions</h3>

<p>In order to precompute a texture we attach it to a framebuffer object (FBO)
and we render a full quad in this FBO. For this we need a basic vertex shader:
*/

namespace csp::atmospheres::models::schneegans::internal {

namespace {

const char kVertexShader[] = R"(
    #version 330
    layout(location = 0) in vec2 vertex;
    void main() {
      gl_Position = vec4(vertex, 0.0, 1.0);
    })";

/*
<p>a basic geometry shader (only for 3D textures, to specify in which layer we
want to write):
*/

const char kGeometryShader[] = R"(
    #version 330
    layout(triangles) in;
    layout(triangle_strip, max_vertices = 3) out;
    uniform int layer;
    void main() {
      gl_Position = gl_in[0].gl_Position;
      gl_Layer = layer;
      EmitVertex();
      gl_Position = gl_in[1].gl_Position;
      gl_Layer = layer;
      EmitVertex();
      gl_Position = gl_in[2].gl_Position;
      gl_Layer = layer;
      EmitVertex();
      EndPrimitive();
    })";

/*
<p>and a fragment shader, which depends on the texture we want to compute. This
is the role of the following shaders, which simply wrap the precomputation
functions from <a href="functions.glsl.html">functions.glsl</a> in complete
shaders (with a <code>main</code> function and a proper declaration of the
shader inputs and outputs). Note that these strings must be concatenated with
<code>definitions.glsl</code> and <code>functions.glsl</code> (provided as C++
string literals by the generated <code>.glsl.inc</code> files), as well as with
a definition of the <code>ATMOSPHERE</code> constant - containing the atmosphere
parameters, to really get a complete shader. Note also the
<code>luminance_from_radiance</code> uniforms: these are used in precomputed
illuminance mode to convert the radiance values computed by the
<code>functions.glsl</code> functions to luminance values (see the
<code>Init</code> method for more details).
*/

const char kComputeTransmittanceShader[] = R"(
    layout(location = 0) out vec3 transmittance;
    void main() {
      transmittance = ComputeTransmittanceToTopAtmosphereBoundaryTexture(
          ATMOSPHERE, gl_FragCoord.xy);
    })";

const char kComputeDirectIrradianceShader[] = R"(
    layout(location = 0) out vec3 delta_irradiance;
    layout(location = 1) out vec3 irradiance;
    uniform sampler2D transmittance_texture;
    void main() {
      delta_irradiance = ComputeDirectIrradianceTexture(
          transmittance_texture, gl_FragCoord.xy);
      irradiance = vec3(0.0);
    })";

const char kComputeSingleScatteringShader[] = R"(
    layout(location = 0) out vec3 delta_rayleigh;
    layout(location = 1) out vec3 delta_mie;
    layout(location = 2) out vec3 accumulated_rayleigh_single_scattering_luminance;
    layout(location = 3) out vec3 accumulated_mie_single_scattering_luminance;
    uniform mat3 luminance_from_radiance;
    uniform sampler2D transmittance_texture;
    uniform int layer;
    void main() {
      ComputeSingleScatteringTexture(
          ATMOSPHERE, transmittance_texture, vec3(gl_FragCoord.xy, layer + 0.5),
          delta_rayleigh, delta_mie);
      accumulated_rayleigh_single_scattering_luminance = luminance_from_radiance * delta_rayleigh;
      accumulated_mie_single_scattering_luminance = luminance_from_radiance * delta_mie;
    })";

const char kComputeScatteringDensityShader[] = R"(
    layout(location = 0) out vec3 scattering_density;
    uniform sampler2D transmittance_texture;
    uniform sampler3D single_rayleigh_scattering_texture;
    uniform sampler3D single_mie_scattering_texture;
    uniform sampler3D multiple_scattering_texture;
    uniform sampler2D irradiance_texture;
    uniform int scattering_order;
    uniform int layer;
    void main() {
      scattering_density = ComputeScatteringDensityTexture(
          ATMOSPHERE, transmittance_texture, single_rayleigh_scattering_texture,
          single_mie_scattering_texture, multiple_scattering_texture,
          irradiance_texture, vec3(gl_FragCoord.xy, layer + 0.5),
          scattering_order);
    })";

const char kComputeIndirectIrradianceShader[] = R"(
    layout(location = 0) out vec3 delta_irradiance;
    layout(location = 1) out vec3 irradiance;
    uniform mat3 luminance_from_radiance;
    uniform sampler3D single_rayleigh_scattering_texture;
    uniform sampler3D single_mie_scattering_texture;
    uniform sampler3D multiple_scattering_texture;
    uniform int scattering_order;
    void main() {
      delta_irradiance = ComputeIndirectIrradianceTexture(
          ATMOSPHERE, single_rayleigh_scattering_texture,
          single_mie_scattering_texture, multiple_scattering_texture,
          gl_FragCoord.xy, scattering_order);
      irradiance = luminance_from_radiance * delta_irradiance;
    })";

const char kComputeMultipleScatteringShader[] = R"(
    layout(location = 0) out vec3 delta_multiple_scattering;
    layout(location = 1) out vec3 multiple_scattering;
    uniform mat3 luminance_from_radiance;
    uniform sampler2D transmittance_texture;
    uniform sampler3D scattering_density_texture;
    uniform int layer;
    void main() {
      float nu;
      delta_multiple_scattering = ComputeMultipleScatteringTexture(
          transmittance_texture, scattering_density_texture,
          vec3(gl_FragCoord.xy, layer + 0.5), nu);
      multiple_scattering = luminance_from_radiance *
                            delta_multiple_scattering / PhaseFunction(ATMOSPHERE.rayleigh, nu);
    })";

/*
<p>We finally need a shader implementing the GLSL functions exposed in our API,
which can be done by calling the corresponding functions in
<a href="functions.glsl.html#rendering">functions.glsl</a>, with the precomputed
texture arguments taken from uniform variables (note also the
*<code>_RADIANCE_TO_LUMINANCE</code> conversion constants in the last functions:
they are computed in the <a href="#utilities">second part</a> below, and their
definitions are concatenated to this GLSL code to get a fully functional
shader).
*/

const char kAtmosphereShader[] = R"(
    uniform sampler2D transmittance_texture;
    uniform sampler3D multiple_scattering_texture;
    uniform sampler3D single_mie_scattering_texture;
    uniform sampler2D irradiance_texture;
    #ifdef RADIANCE_API_ENABLED
    RadianceSpectrum GetSolarRadiance() {
      return SOLAR_IRRADIANCE /
          (PI * SUN_ANGULAR_RADIUS * SUN_ANGULAR_RADIUS);
    }
    RadianceSpectrum GetSkyRadiance(
        Position camera, Direction view_ray,
        Direction sun_direction, out DimensionlessSpectrum transmittance) {
      return GetSkyRadiance(ATMOSPHERE, transmittance_texture,
          multiple_scattering_texture, single_mie_scattering_texture,
          camera, view_ray, 0.0, sun_direction, transmittance);
    }
    RadianceSpectrum GetSkyRadianceToPoint(
        Position camera, Position point,
        Direction sun_direction, out DimensionlessSpectrum transmittance) {
      return GetSkyRadianceToPoint(ATMOSPHERE, transmittance_texture,
          multiple_scattering_texture, single_mie_scattering_texture,
          camera, point, 0.0, sun_direction, transmittance);
    }
    IrradianceSpectrum GetSunAndSkyIrradiance(
       Position p, Direction sun_direction,
       out IrradianceSpectrum sky_irradiance) {
      return GetSunAndSkyIrradiance(transmittance_texture,
          irradiance_texture, p, sun_direction, sky_irradiance);
    }
    #endif
    Luminance3 GetSolarLuminance() {
      return SOLAR_IRRADIANCE /
          (PI * SUN_ANGULAR_RADIUS * SUN_ANGULAR_RADIUS) *
          SUN_SPECTRAL_RADIANCE_TO_LUMINANCE;
    }
    Luminance3 GetSkyLuminance(
        Position camera, Direction view_ray,
        Direction sun_direction, out DimensionlessSpectrum transmittance) {
      return GetSkyRadiance(ATMOSPHERE, transmittance_texture,
          multiple_scattering_texture, single_mie_scattering_texture,
          camera, view_ray, 0.0, sun_direction, transmittance) *
          SKY_SPECTRAL_RADIANCE_TO_LUMINANCE;
    }
    Luminance3 GetSkyLuminanceToPoint(
        Position camera, Position point,
        Direction sun_direction, out DimensionlessSpectrum transmittance) {
      return GetSkyRadianceToPoint(ATMOSPHERE, transmittance_texture,
          multiple_scattering_texture, single_mie_scattering_texture,
          camera, point, 0.0, sun_direction, transmittance) *
          SKY_SPECTRAL_RADIANCE_TO_LUMINANCE;
    }
    Illuminance3 GetSunAndSkyIlluminance(
       Position p, Direction sun_direction,
       out IrradianceSpectrum sky_irradiance) {
      IrradianceSpectrum sun_irradiance = GetSunAndSkyIrradiance(
          transmittance_texture, irradiance_texture, p,
          sun_direction, sky_irradiance);
      sky_irradiance *= SKY_SPECTRAL_RADIANCE_TO_LUMINANCE;
      return sun_irradiance * SUN_SPECTRAL_RADIANCE_TO_LUMINANCE;
    })";

/*<h3 id="utilities">Utility classes and functions</h3>

<p>To compile and link these shaders into programs, and to set their uniforms,
we use the following utility class:
*/

class Program {
 public:
  Program(const std::string& vertex_shader_source, const std::string& fragment_shader_source)
      : Program(vertex_shader_source, "", fragment_shader_source) {
  }

  Program(const std::string& vertex_shader_source, const std::string& geometry_shader_source,
      const std::string& fragment_shader_source) {
    program_ = glCreateProgram();

    const char* source;
    source               = vertex_shader_source.c_str();
    GLuint vertex_shader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertex_shader, 1, &source, NULL);
    glCompileShader(vertex_shader);
    CheckShader(vertex_shader);
    glAttachShader(program_, vertex_shader);

    GLuint geometry_shader = 0;
    if (!geometry_shader_source.empty()) {
      source          = geometry_shader_source.c_str();
      geometry_shader = glCreateShader(GL_GEOMETRY_SHADER);
      glShaderSource(geometry_shader, 1, &source, NULL);
      glCompileShader(geometry_shader);
      CheckShader(geometry_shader);
      glAttachShader(program_, geometry_shader);
    }

    source                 = fragment_shader_source.c_str();
    GLuint fragment_shader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragment_shader, 1, &source, NULL);
    glCompileShader(fragment_shader);
    CheckShader(fragment_shader);
    glAttachShader(program_, fragment_shader);

    glLinkProgram(program_);
    CheckProgram(program_);

    glDetachShader(program_, vertex_shader);
    glDeleteShader(vertex_shader);
    if (!geometry_shader_source.empty()) {
      glDetachShader(program_, geometry_shader);
      glDeleteShader(geometry_shader);
    }
    glDetachShader(program_, fragment_shader);
    glDeleteShader(fragment_shader);
  }

  ~Program() {
    glDeleteProgram(program_);
  }

  void Use() const {
    glUseProgram(program_);
  }

  void BindMat3(const std::string& uniform_name, const std::array<float, 9>& value) const {
    glUniformMatrix3fv(glGetUniformLocation(program_, uniform_name.c_str()), 1,
        true /* transpose */, value.data());
  }

  void BindInt(const std::string& uniform_name, int value) const {
    glUniform1i(glGetUniformLocation(program_, uniform_name.c_str()), value);
  }

  void BindTexture2d(
      const std::string& sampler_uniform_name, GLuint texture, GLuint texture_unit) const {
    glActiveTexture(GL_TEXTURE0 + texture_unit);
    glBindTexture(GL_TEXTURE_2D, texture);
    BindInt(sampler_uniform_name, texture_unit);
  }

  void BindTexture3d(
      const std::string& sampler_uniform_name, GLuint texture, GLuint texture_unit) const {
    glActiveTexture(GL_TEXTURE0 + texture_unit);
    glBindTexture(GL_TEXTURE_3D, texture);
    BindInt(sampler_uniform_name, texture_unit);
  }

 private:
  static void CheckShader(GLuint shader) {
    GLint compile_status;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &compile_status);
    if (compile_status == GL_FALSE) {
      PrintShaderLog(shader);
    }
    assert(compile_status == GL_TRUE);
  }

  static void PrintShaderLog(GLuint shader) {
    GLint log_length;
    glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &log_length);
    if (log_length > 0) {
      std::unique_ptr<char[]> log_data(new char[log_length]);
      glGetShaderInfoLog(shader, log_length, &log_length, log_data.get());
      std::cerr << "compile log = " << std::string(log_data.get(), log_length) << std::endl;
    }
  }

  static void CheckProgram(GLuint program) {
    GLint link_status;
    glGetProgramiv(program, GL_LINK_STATUS, &link_status);
    if (link_status == GL_FALSE) {
      PrintProgramLog(program);
    }
    assert(link_status == GL_TRUE);
    assert(glGetError() == 0);
  }

  static void PrintProgramLog(GLuint program) {
    GLint log_length;
    glGetProgramiv(program, GL_INFO_LOG_LENGTH, &log_length);
    if (log_length > 0) {
      std::unique_ptr<char[]> log_data(new char[log_length]);
      glGetProgramInfoLog(program, log_length, &log_length, log_data.get());
      std::cerr << "link log = " << std::string(log_data.get(), log_length) << std::endl;
    }
  }

  GLuint program_;
};

/*
<p>We also need functions to allocate the precomputed textures on GPU:
*/

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

/*
<p>and a function to draw a full screen quad in an offscreen framebuffer (with
blending separately enabled or disabled for each color attachment):
*/

void DrawQuad(const std::vector<bool>& enable_blend, GLuint quad_vao) {
  for (unsigned int i = 0; i < enable_blend.size(); ++i) {
    if (enable_blend[i]) {
      glEnablei(GL_BLEND, i);
    }
  }

  glBindVertexArray(quad_vao);
  glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
  glBindVertexArray(0);

  for (unsigned int i = 0; i < enable_blend.size(); ++i) {
    glDisablei(GL_BLEND, i);
  }
}

/*
<p>Finally, we need a utility function to compute the value of the conversion
constants *<code>_RADIANCE_TO_LUMINANCE</code>, used above to convert the
spectral results into luminance values. These are the constants k_r, k_g, k_b
described in Section 14.3 of <a href="https://arxiv.org/pdf/1612.04336.pdf">A
Qualitative and Quantitative Evaluation of 8 Clear Sky Models</a>.

<p>Computing their value requires an integral of a function times a CIE color
matching function. Thus, we first need functions to interpolate an arbitrary
function (specified by some samples), and a CIE color matching function
(specified by tabulated values), at an arbitrary wavelength. This is the purpose
of the following two functions:
*/

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

double Interpolate(const std::vector<double>& xVals, const std::vector<double>& yVals, double x) {
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

/*
<p>We can then implement a utility function to compute the "spectral radiance to
luminance" conversion constants (see Section 14.3 in <a
href="https://arxiv.org/pdf/1612.04336.pdf">A Qualitative and Quantitative
Evaluation of 8 Clear Sky Models</a> for their definitions):
*/

// The returned constants are in lumen.nm / watt.
void ComputeSpectralRadianceToLuminanceFactors(
    double lambda_power, double* k_r, double* k_g, double* k_b) {
  *k_r           = 0.0;
  *k_g           = 0.0;
  *k_b           = 0.0;
  double solar_r = Interpolate(WAVELENGTHS, SOLAR_IRRADIANCE, Model::kLambdaR);
  double solar_g = Interpolate(WAVELENGTHS, SOLAR_IRRADIANCE, Model::kLambdaG);
  double solar_b = Interpolate(WAVELENGTHS, SOLAR_IRRADIANCE, Model::kLambdaB);
  int    dlambda = 1;
  for (int lambda = WAVELENGTHS.front(); lambda <= WAVELENGTHS.back(); lambda += dlambda) {
    double        x_bar      = CieColorMatchingFunctionTableValue(lambda, 1);
    double        y_bar      = CieColorMatchingFunctionTableValue(lambda, 2);
    double        z_bar      = CieColorMatchingFunctionTableValue(lambda, 3);
    const double* xyz2srgb   = XYZ_TO_SRGB;
    double        r_bar      = xyz2srgb[0] * x_bar + xyz2srgb[1] * y_bar + xyz2srgb[2] * z_bar;
    double        g_bar      = xyz2srgb[3] * x_bar + xyz2srgb[4] * y_bar + xyz2srgb[5] * z_bar;
    double        b_bar      = xyz2srgb[6] * x_bar + xyz2srgb[7] * y_bar + xyz2srgb[8] * z_bar;
    double        irradiance = Interpolate(WAVELENGTHS, SOLAR_IRRADIANCE, lambda);
    *k_r += r_bar * irradiance / solar_r * pow(lambda / Model::kLambdaR, lambda_power);
    *k_g += g_bar * irradiance / solar_g * pow(lambda / Model::kLambdaG, lambda_power);
    *k_b += b_bar * irradiance / solar_b * pow(lambda / Model::kLambdaB, lambda_power);
  }
  *k_r *= MAX_LUMINOUS_EFFICACY * dlambda;
  *k_g *= MAX_LUMINOUS_EFFICACY * dlambda;
  *k_b *= MAX_LUMINOUS_EFFICACY * dlambda;
}

} // anonymous namespace

/*<h3 id="implementation">Model implementation</h3>

<p>Using the above utility functions and classes, we can now implement the
constructor of the <code>Model</code> class. This constructor generates a piece
of GLSL code that defines an <code>ATMOSPHERE</code> constant containing the
atmosphere parameters (we use constants instead of uniforms to enable constant
folding and propagation optimizations in the GLSL compiler), concatenated with
<a href="functions.glsl.html">functions.glsl</a>, and with
<code>kAtmosphereShader</code>, to get the shader exposed by our API in
<code>GetShader</code>. It also allocates the precomputed textures (but does not
initialize them), as well as a vertex buffer object to render a full screen quad
(used to render into the precomputed textures).
*/

Model::Model(const std::vector<double>& wavelengths, const double sun_angular_radius,
    double bottom_radius, double top_radius, const ScatteringAtmosphereComponent& rayleigh,
    const ScatteringAtmosphereComponent& mie, const AbsorbingAtmosphereComponent& ozone,
    double ground_albedo, double max_sun_zenith_angle, double length_unit_in_meters)
    : wavelengths_(wavelengths)
    , rayleigh_(rayleigh)
    , mie_(mie)
    , ozone_(ozone) {

  auto extractVec3 = [](const std::vector<double>& xVals, const std::vector<double>& yVals,
                         const vec3& lambdas, double scale = 1.0) {
    double r = Interpolate(xVals, yVals, lambdas[0]) * scale;
    double g = Interpolate(xVals, yVals, lambdas[1]) * scale;
    double b = Interpolate(xVals, yVals, lambdas[2]) * scale;
    return "vec3(" + cs::utils::toString(r) + "," + cs::utils::toString(g) + "," +
           cs::utils::toString(b) + ")";
  };

  auto scatteringComponent = [this, extractVec3, length_unit_in_meters](
                                 ScatteringAtmosphereComponent const& component,
                                 float phaseTextureV, float densityTextureV, vec3 lambdas) {
    std::stringstream ss;
    ss << "ScatteringComponent(";

    ss << phaseTextureV << ",\n";
    ss << densityTextureV << ",\n";

    auto absorption =
        extractVec3(wavelengths_, component.absorption, lambdas, length_unit_in_meters);
    auto scattering =
        extractVec3(wavelengths_, component.scattering, lambdas, length_unit_in_meters);

    ss << scattering << " + " << absorption << ",\n";
    ss << scattering << "\n";

    ss << ")";

    return ss.str();
  };

  auto absorbingComponent = [this, extractVec3, length_unit_in_meters](
                                AbsorbingAtmosphereComponent const& component,
                                float densityTextureV, vec3 lambdas) {
    std::stringstream ss;
    ss << "AbsorbingComponent(";

    ss << densityTextureV << ",\n";

    auto absorption =
        extractVec3(wavelengths_, component.absorption, lambdas, length_unit_in_meters);

    ss << absorption << "\n";

    ss << ")";

    return ss.str();
  };

  // Compute the values for the SKY_RADIANCE_TO_LUMINANCE constant. In theory
  // this should be 1 in precomputed illuminance mode (because the precomputed
  // textures already contain illuminance values). In practice, however, storing
  // true illuminance values in half precision textures yields artefacts
  // (because the values are too large), so we store illuminance values divided
  // by MAX_LUMINOUS_EFFICACY instead. This is why, in precomputed illuminance
  // mode, we set SKY_RADIANCE_TO_LUMINANCE to MAX_LUMINOUS_EFFICACY.
  bool   precompute_illuminance = wavelengths_.size() > 3;
  double sky_k_r, sky_k_g, sky_k_b;
  if (precompute_illuminance) {
    sky_k_r = sky_k_g = sky_k_b = MAX_LUMINOUS_EFFICACY;
  } else {
    ComputeSpectralRadianceToLuminanceFactors(-3 /* lambda_power */, &sky_k_r, &sky_k_g, &sky_k_b);
  }
  // Compute the values for the SUN_RADIANCE_TO_LUMINANCE constant.
  double sun_k_r, sun_k_g, sun_k_b;
  ComputeSpectralRadianceToLuminanceFactors(0 /* lambda_power */, &sun_k_r, &sun_k_g, &sun_k_b);

  // A lambda that creates a GLSL header containing our atmosphere computation
  // functions, specialized for the given atmosphere parameters and for the 3
  // wavelengths in 'lambdas'.
  auto definitions_glsl = cs::utils::filesystem::loadToString(
      "../share/resources/shaders/csp-atmospheres/models/schneegans/definitions.glsl");
  auto functions_glsl = cs::utils::filesystem::loadToString(
      "../share/resources/shaders/csp-atmospheres/models/schneegans/functions.glsl");

  // clang-format off
  glsl_header_factory_ = [=](const vec3& lambdas) {
    return "#version 330\n"
            "#define IN(x) const in x\n"
            "#define OUT(x) out x\n"
            "#define TEMPLATE(x)\n"
            "#define TEMPLATE_ARGUMENT(x)\n"
            "#define assert(x)\n"
            "const int TRANSMITTANCE_TEXTURE_WIDTH = "  + cs::utils::toString(TRANSMITTANCE_TEXTURE_WIDTH) + ";\n" +
            "const int TRANSMITTANCE_TEXTURE_HEIGHT = " + cs::utils::toString(TRANSMITTANCE_TEXTURE_HEIGHT) + ";\n" +
            "const int SCATTERING_TEXTURE_R_SIZE = "    + cs::utils::toString(SCATTERING_TEXTURE_R_SIZE) + ";\n" +
            "const int SCATTERING_TEXTURE_MU_SIZE = "   + cs::utils::toString(SCATTERING_TEXTURE_MU_SIZE) + ";\n" +
            "const int SCATTERING_TEXTURE_MU_S_SIZE = " + cs::utils::toString(SCATTERING_TEXTURE_MU_S_SIZE) + ";\n" +
            "const int SCATTERING_TEXTURE_NU_SIZE = "   + cs::utils::toString(SCATTERING_TEXTURE_NU_SIZE) + ";\n" +
            "const int IRRADIANCE_TEXTURE_WIDTH = "     + cs::utils::toString(IRRADIANCE_TEXTURE_WIDTH) + ";\n" +
            "const int IRRADIANCE_TEXTURE_HEIGHT = "    + cs::utils::toString(IRRADIANCE_TEXTURE_HEIGHT) + ";\n" +
            "const int SAMPLE_COUNT_OPTICAL_DEPTH = "   + cs::utils::toString(SAMPLE_COUNT_OPTICAL_DEPTH) + ";\n" +
            "const int SAMPLE_COUNT_SINGLE_SCATTERING = "   + cs::utils::toString(SAMPLE_COUNT_SINGLE_SCATTERING) + ";\n" +
            "const int SAMPLE_COUNT_SCATTERING_DENSITY = "  + cs::utils::toString(SAMPLE_COUNT_SCATTERING_DENSITY) + ";\n" +
            "const int SAMPLE_COUNT_MULTI_SCATTERING = "    + cs::utils::toString(SAMPLE_COUNT_MULTI_SCATTERING) + ";\n" +
            "const int SAMPLE_COUNT_INDIRECT_IRRADIANCE = " + cs::utils::toString(SAMPLE_COUNT_INDIRECT_IRRADIANCE) + ";\n" +
            definitions_glsl +
            "const vec3 SKY_SPECTRAL_RADIANCE_TO_LUMINANCE = vec3(" + cs::utils::toString(sky_k_r) + "," + cs::utils::toString(sky_k_g) + "," + cs::utils::toString(sky_k_b) + ");\n" +
            "const vec3 SUN_SPECTRAL_RADIANCE_TO_LUMINANCE = vec3(" + cs::utils::toString(sun_k_r) + "," + cs::utils::toString(sun_k_g) + "," + cs::utils::toString(sun_k_b) + ");\n" +
            "const vec3 SOLAR_IRRADIANCE = "            + extractVec3(WAVELENGTHS, SOLAR_IRRADIANCE, lambdas) + ";\n" +
            "const vec3 GROUND_ALBEDO = vec3("          + cs::utils::toString(ground_albedo) + ");\n" +
            "const float SUN_ANGULAR_RADIUS = "         + cs::utils::toString(sun_angular_radius) + ";\n" +
            "const float BOTTOM_RADIUS = "              + cs::utils::toString(bottom_radius / length_unit_in_meters) + ";\n" +
            "const float TOP_RADIUS = "                 + cs::utils::toString(top_radius / length_unit_in_meters) + ";\n" +
            "const float MU_S_MIN = "                   + cs::utils::toString(cos(max_sun_zenith_angle))+ ";\n" +
            "const AtmosphereComponents ATMOSPHERE = AtmosphereComponents(\n" +
              scatteringComponent(rayleigh_, 0.0, 0.0, lambdas) + ",\n" +
              scatteringComponent(mie_, 1.0, 0.5, lambdas) + ",\n" +
              absorbingComponent(ozone_, 1.0, lambdas) + ");\n" +
            functions_glsl;
    };
  // clang-format on

  // Allocate the precomputed textures, but don't precompute them yet.
  transmittance_texture_ = NewTexture2d(
      TRANSMITTANCE_TEXTURE_WIDTH, TRANSMITTANCE_TEXTURE_HEIGHT, GL_RGBA32F, GL_RGBA, GL_FLOAT);
  multiple_scattering_texture_   = NewTexture3d(SCATTERING_TEXTURE_WIDTH, SCATTERING_TEXTURE_HEIGHT,
      SCATTERING_TEXTURE_DEPTH, GL_RGBA32F, GL_RGBA, GL_FLOAT);
  single_mie_scattering_texture_ = NewTexture3d(SCATTERING_TEXTURE_WIDTH, SCATTERING_TEXTURE_HEIGHT,
      SCATTERING_TEXTURE_DEPTH, GL_RGBA32F, GL_RGBA, GL_FLOAT);

  irradiance_texture_ = NewTexture2d(
      IRRADIANCE_TEXTURE_WIDTH, IRRADIANCE_TEXTURE_HEIGHT, GL_RGBA32F, GL_RGBA, GL_FLOAT);

  // Create the density profile texture.
  {

    size_t numDensities = rayleigh_.density.size();

    std::vector<float> densityData;
    size_t             numComponents = 3; // Rayleigh, Mie, Ozone
    densityData.reserve(numComponents * numDensities);

    densityData.insert(densityData.end(), rayleigh_.density.begin(), rayleigh_.density.end());
    densityData.insert(densityData.end(), mie_.density.begin(), mie_.density.end());
    densityData.insert(densityData.end(), ozone_.density.begin(), ozone_.density.end());

    density_texture_ =
        NewTexture2d(numDensities, numComponents, GL_R32F, GL_RED, GL_FLOAT, densityData.data());
  }

  // Create and compile the shader providing our API.
  std::string shader = glsl_header_factory_({kLambdaR, kLambdaG, kLambdaB}) +
                       (precompute_illuminance ? "" : "#define RADIANCE_API_ENABLED\n") +
                       kAtmosphereShader;
  const char* source = shader.c_str();
  atmosphere_shader_ = glCreateShader(GL_FRAGMENT_SHADER);
  glShaderSource(atmosphere_shader_, 1, &source, NULL);
  glCompileShader(atmosphere_shader_);

  // Create a full screen quad vertex array and vertex buffer objects.
  glGenVertexArrays(1, &full_screen_quad_vao_);
  glBindVertexArray(full_screen_quad_vao_);
  glGenBuffers(1, &full_screen_quad_vbo_);
  glBindBuffer(GL_ARRAY_BUFFER, full_screen_quad_vbo_);
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

/*
<p>The destructor is trivial:
*/

Model::~Model() {
  glDeleteBuffers(1, &full_screen_quad_vbo_);
  glDeleteVertexArrays(1, &full_screen_quad_vao_);
  glDeleteTextures(1, &phase_texture_);
  glDeleteTextures(1, &transmittance_texture_);
  glDeleteTextures(1, &multiple_scattering_texture_);
  glDeleteTextures(1, &single_mie_scattering_texture_);
  glDeleteTextures(1, &irradiance_texture_);
  glDeleteShader(atmosphere_shader_);
}

/*
<p>The Init method precomputes the atmosphere textures. It first allocates the
temporary resources it needs, then calls <code>Precompute</code> to do the
actual precomputations, and finally destroys the temporary resources.

<p>Note that there are two precomputation modes here, depending on whether we
want to store precomputed irradiance or illuminance values:
<ul>
  <li>In precomputed irradiance mode, we simply need to call
  <code>Precompute</code> with the 3 wavelengths for which we want to precompute
  irradiance, namely <code>kLambdaR</code>, <code>kLambdaG</code>,
  <code>kLambdaB</code> (with the identity matrix for
  <code>luminance_from_radiance</code>, since we don't want any conversion from
  radiance to luminance)</li>
  <li>In precomputed illuminance mode, we need to precompute irradiance for
  <code>num_precomputed_wavelengths_</code>, and then integrate the results,
  multiplied with the 3 CIE xyz color matching functions and the XYZ to sRGB
  matrix to get sRGB illuminance values.
  <p>A naive solution would be to allocate temporary textures for the
  intermediate irradiance results, then perform the integration from irradiance
  to illuminance and store the result in the final precomputed texture. In
  pseudo-code (and assuming one wavelength per texture instead of 3):
  <pre>
    create n temporary irradiance textures
    for each wavelength lambda in the n wavelengths:
       precompute irradiance at lambda into one of the temporary textures
    initializes the final illuminance texture with zeros
    for each wavelength lambda in the n wavelengths:
      accumulate in the final illuminance texture the product of the
      precomputed irradiance at lambda (read from the temporary textures)
      with the value of the 3 sRGB color matching functions at lambda (i.e.
      the product of the XYZ to sRGB matrix with the CIE xyz color matching
      functions).
  </pre>
  <p>However, this be would waste GPU memory. Instead, we can avoid allocating
  temporary irradiance textures, by merging the two above loops:
  <pre>
    for each wavelength lambda in the n wavelengths:
      accumulate in the final illuminance texture (or, for the first
      iteration, set this texture to) the product of the precomputed
      irradiance at lambda (computed on the fly) with the value of the 3
      sRGB color matching functions at lambda.
  </pre>
  <p>This is the method we use below, with 3 wavelengths per iteration instead
  of 1, using <code>Precompute</code> to compute 3 irradiances values per
  iteration, and <code>luminance_from_radiance</code> to multiply 3 irradiances
  with the values of the 3 sRGB color matching functions at 3 different
  wavelengths (yielding a 3x3 matrix).</li>
</ul>

<p>This yields the following implementation:
*/

void Model::Init(unsigned int num_scattering_orders) {
  // The precomputations require temporary textures, in particular to store the
  // contribution of one scattering order, which is needed to compute the next
  // order of scattering (the final precomputed textures store the sum of all
  // the scattering orders). We allocate them here, and destroy them at the end
  // of this method.
  GLuint delta_irradiance_texture = NewTexture2d(
      IRRADIANCE_TEXTURE_WIDTH, IRRADIANCE_TEXTURE_HEIGHT, GL_RGBA32F, GL_RGBA, GL_FLOAT);
  GLuint delta_rayleigh_scattering_texture = NewTexture3d(SCATTERING_TEXTURE_WIDTH,
      SCATTERING_TEXTURE_HEIGHT, SCATTERING_TEXTURE_DEPTH, GL_RGBA32F, GL_RGBA, GL_FLOAT);
  GLuint delta_mie_scattering_texture      = NewTexture3d(SCATTERING_TEXTURE_WIDTH,
      SCATTERING_TEXTURE_HEIGHT, SCATTERING_TEXTURE_DEPTH, GL_RGBA32F, GL_RGBA, GL_FLOAT);
  GLuint delta_scattering_density_texture  = NewTexture3d(SCATTERING_TEXTURE_WIDTH,
      SCATTERING_TEXTURE_HEIGHT, SCATTERING_TEXTURE_DEPTH, GL_RGBA32F, GL_RGBA, GL_FLOAT);
  // delta_multiple_scattering_texture is only needed to compute scattering
  // order 3 or more, while delta_rayleigh_scattering_texture and
  // delta_mie_scattering_texture are only needed to compute double scattering.
  // Therefore, to save memory, we can store delta_rayleigh_scattering_texture
  // and delta_multiple_scattering_texture in the same GPU texture.
  GLuint delta_multiple_scattering_texture = delta_rayleigh_scattering_texture;

  // The precomputations also require a temporary framebuffer object, created
  // here (and destroyed at the end of this method).
  GLuint fbo;
  glGenFramebuffers(1, &fbo);
  glBindFramebuffer(GL_FRAMEBUFFER, fbo);

  // The actual precomputations depend on whether we want to store precomputed
  // irradiance or illuminance values.
  if (wavelengths_.size() <= 3) {
    logger().info("Precomputing atmospheric scattering...");
    vec3 lambdas{kLambdaR, kLambdaG, kLambdaB};
    mat3 luminance_from_radiance{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
    Precompute(fbo, delta_irradiance_texture, delta_rayleigh_scattering_texture,
        delta_mie_scattering_texture, delta_scattering_density_texture,
        delta_multiple_scattering_texture, lambdas, luminance_from_radiance, false /* blend */,
        num_scattering_orders);
  } else {
    int num_iterations = wavelengths_.size() / 3;
    for (int i = 0; i < num_iterations; ++i) {
      logger().info("Precomputing atmospheric scattering ({}/{})...", i + 1, num_iterations);

      vec3 lambdas{wavelengths_[i * 3 + 0], wavelengths_[i * 3 + 1], wavelengths_[i * 3 + 2]};
      auto coeff = [this](double lambda, int component) {
        // Note that we don't include MAX_LUMINOUS_EFFICACY here, to avoid
        // artefacts due to too large values when using half precision on GPU.
        // We add this term back in kAtmosphereShader, via
        // SKY_SPECTRAL_RADIANCE_TO_LUMINANCE (see also the comments in the
        // Model constructor).
        double x = CieColorMatchingFunctionTableValue(lambda, 1);
        double y = CieColorMatchingFunctionTableValue(lambda, 2);
        double z = CieColorMatchingFunctionTableValue(lambda, 3);
        return static_cast<float>(
            (XYZ_TO_SRGB[component * 3] * x + XYZ_TO_SRGB[component * 3 + 1] * y +
                XYZ_TO_SRGB[component * 3 + 2] * z) *
            (wavelengths_[1] - wavelengths_[0]));
      };

      mat3 luminance_from_radiance{coeff(lambdas[0], 0), coeff(lambdas[1], 0), coeff(lambdas[2], 0),
          coeff(lambdas[0], 1), coeff(lambdas[1], 1), coeff(lambdas[2], 1), coeff(lambdas[0], 2),
          coeff(lambdas[1], 2), coeff(lambdas[2], 2)};

      Precompute(fbo, delta_irradiance_texture, delta_rayleigh_scattering_texture,
          delta_mie_scattering_texture, delta_scattering_density_texture,
          delta_multiple_scattering_texture, lambdas, luminance_from_radiance, i > 0 /* blend */,
          num_scattering_orders);
    }

    // After the above iterations, the transmittance texture contains the
    // transmittance for the 3 wavelengths used at the last iteration. But we
    // want the transmittance at kLambdaR, kLambdaG, kLambdaB instead, so we
    // must recompute it here for these 3 wavelengths:
    std::string header = glsl_header_factory_({kLambdaR, kLambdaG, kLambdaB});
    Program     compute_transmittance(kVertexShader, header + kComputeTransmittanceShader);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, transmittance_texture_, 0);
    glDrawBuffer(GL_COLOR_ATTACHMENT0);
    glViewport(0, 0, TRANSMITTANCE_TEXTURE_WIDTH, TRANSMITTANCE_TEXTURE_HEIGHT);
    glScissor(0, 0, TRANSMITTANCE_TEXTURE_WIDTH, TRANSMITTANCE_TEXTURE_HEIGHT);
    compute_transmittance.Use();
    compute_transmittance.BindTexture2d("density_texture", density_texture_, 0);
    DrawQuad({false}, full_screen_quad_vao_);

    glFlush();

    // Also, the phase_texture_ contains the phase functions for the last used wavelengths. We need
    // to update it with kLambdaR, kLambdaG, kLambdaB as well.
    UpdatePhaseFunctionTexture({rayleigh_, mie_}, {kLambdaR, kLambdaG, kLambdaB});
  }

  // Delete the temporary resources allocated at the begining of this method.
  glUseProgram(0);
  glBindFramebuffer(GL_FRAMEBUFFER, 0);
  glDeleteFramebuffers(1, &fbo);
  glDeleteTextures(1, &density_texture_);
  glDeleteTextures(1, &delta_scattering_density_texture);
  glDeleteTextures(1, &delta_mie_scattering_texture);
  glDeleteTextures(1, &delta_rayleigh_scattering_texture);
  glDeleteTextures(1, &delta_irradiance_texture);
  assert(glGetError() == 0);
}

/*
<p>The <code>SetProgramUniforms</code> method is straightforward: it simply
binds the precomputed textures to the specified texture units, and then sets
the corresponding uniforms in the user provided program to the index of these
texture units.
*/

void Model::SetProgramUniforms(GLuint program, GLuint phase_texture_unit,
    GLuint transmittance_texture_unit, GLuint multiple_scattering_texture_unit,
    GLuint irradiance_texture_unit, GLuint single_mie_scattering_texture_unit) const {

  glActiveTexture(GL_TEXTURE0 + phase_texture_unit);
  glBindTexture(GL_TEXTURE_2D, phase_texture_);
  glUniform1i(glGetUniformLocation(program, "phase_texture"), phase_texture_unit);

  glActiveTexture(GL_TEXTURE0 + transmittance_texture_unit);
  glBindTexture(GL_TEXTURE_2D, transmittance_texture_);
  glUniform1i(glGetUniformLocation(program, "transmittance_texture"), transmittance_texture_unit);

  glActiveTexture(GL_TEXTURE0 + multiple_scattering_texture_unit);
  glBindTexture(GL_TEXTURE_3D, multiple_scattering_texture_);
  glUniform1i(glGetUniformLocation(program, "multiple_scattering_texture"),
      multiple_scattering_texture_unit);

  glActiveTexture(GL_TEXTURE0 + irradiance_texture_unit);
  glBindTexture(GL_TEXTURE_2D, irradiance_texture_);
  glUniform1i(glGetUniformLocation(program, "irradiance_texture"), irradiance_texture_unit);

  glActiveTexture(GL_TEXTURE0 + single_mie_scattering_texture_unit);
  glBindTexture(GL_TEXTURE_3D, single_mie_scattering_texture_);
  glUniform1i(glGetUniformLocation(program, "single_mie_scattering_texture"),
      single_mie_scattering_texture_unit);
}

/*
Here is an outline of the data flow of the precomputation.

1. Compute the transmittance (vec3) of the atmosphere for every point in every direction and store
   it in transmittance_texture_. This incorporates extinction based on rayleigh, mie, and ozone
   particles.

2. Using transmittance_texture_, compute the direct irradiance from the Sun to every point in the
   atmosphere for the current set of wavelengths and store it in delta_irradiance_texture. In this
   step, irradiance_texture_ os also initialized to zero if it's the first call to Precompute().

3. Using the transmittance_texture_, compute the single mie and single rayleigh scattering
   irradiance along the rays in the atmosphere. This is the rayleigh and mie density * solar
   irradiance * scattering coefficient. The term stored in the output textures is without the phase
   function. The irradiance for the current set of wavelengths is stored in
   delta_rayleigh_scattering_texture and delta_mie_scattering_texture. It is also converted to
   illuminance and accumulated for all wavelengths in multiple_scattering_texture_ and
   single_mie_scattering_texture_.

At this point, multiple_scattering_texture_ and single_mie_scattering_texture_ contain single
scattering illuminance without the phase function.

4. Iteratively compute higher orders of scattering. The following happens in a loop:

   4.1. Compute the scattering density, and store it in delta_scattering_density_texture.

   4.2. Compute the indirect irradiance, store it in delta_irradiance_texture and accumulate it in
        irradiance_texture_.

   4.3. Compute the multiple scattering, store it in delta_multiple_scattering_texture, and
        accumulate it in multiple_scattering_texture_.

At the end, single_mie_scattering_texture_ contains the single mie scattering illuminance
without the phase function and multiple_scattering_texture_ contains single rayleigh scattering
without the phase function + multiple scattering with only the mie phase function applied. So at
render time, the data from single_mie_scattering_texture_ needs to be multiplied with the mie phase
function and the data from multiple_scattering_texture_ needs to be multiplied with the rayleigh
phase function.

*/
void Model::Precompute(GLuint fbo, GLuint delta_irradiance_texture,
    GLuint delta_rayleigh_scattering_texture, GLuint delta_mie_scattering_texture,
    GLuint delta_scattering_density_texture, GLuint delta_multiple_scattering_texture,
    const vec3& lambdas, const mat3& luminance_from_radiance, bool blend,
    unsigned int num_scattering_orders) {
  // The precomputations require specific GLSL programs, for each precomputation
  // step. We create and compile them here (they are automatically destroyed
  // when this method returns, via the Program destructor).
  std::string header = glsl_header_factory_(lambdas);

  UpdatePhaseFunctionTexture({rayleigh_, mie_}, lambdas);

  Program compute_transmittance(kVertexShader, header + kComputeTransmittanceShader);
  Program compute_direct_irradiance(kVertexShader, header + kComputeDirectIrradianceShader);
  Program compute_single_scattering(
      kVertexShader, kGeometryShader, header + kComputeSingleScatteringShader);
  Program compute_scattering_density(
      kVertexShader, kGeometryShader, header + kComputeScatteringDensityShader);
  Program compute_indirect_irradiance(kVertexShader, header + kComputeIndirectIrradianceShader);
  Program compute_multiple_scattering(
      kVertexShader, kGeometryShader, header + kComputeMultipleScatteringShader);

  const GLuint kDrawBuffers[4] = {
      GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1, GL_COLOR_ATTACHMENT2, GL_COLOR_ATTACHMENT3};
  glBlendEquationSeparate(GL_FUNC_ADD, GL_FUNC_ADD);
  glBlendFuncSeparate(GL_ONE, GL_ONE, GL_ONE, GL_ONE);

  // -----------------------------------------------------------------------------------------------

  // 1. Compute the transmittance, and store it in transmittance_texture_.
  glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, transmittance_texture_, 0);
  glDrawBuffer(GL_COLOR_ATTACHMENT0);
  glViewport(0, 0, TRANSMITTANCE_TEXTURE_WIDTH, TRANSMITTANCE_TEXTURE_HEIGHT);
  glScissor(0, 0, TRANSMITTANCE_TEXTURE_WIDTH, TRANSMITTANCE_TEXTURE_HEIGHT);
  compute_transmittance.Use();
  compute_transmittance.BindTexture2d("density_texture", density_texture_, 0);
  DrawQuad({false}, full_screen_quad_vao_);

  // -----------------------------------------------------------------------------------------------

  // 2. Compute the direct irradiance, store it in delta_irradiance_texture and,
  // depending on 'blend', either initialize irradiance_texture_ with zeros or
  // leave it unchanged (we don't want the direct irradiance in
  // irradiance_texture_, but only the irradiance from the sky).
  glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, delta_irradiance_texture, 0);
  glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, irradiance_texture_, 0);
  glDrawBuffers(2, kDrawBuffers);
  glViewport(0, 0, IRRADIANCE_TEXTURE_WIDTH, IRRADIANCE_TEXTURE_HEIGHT);
  glScissor(0, 0, IRRADIANCE_TEXTURE_WIDTH, IRRADIANCE_TEXTURE_HEIGHT);
  compute_direct_irradiance.Use();
  compute_direct_irradiance.BindTexture2d("transmittance_texture", transmittance_texture_, 0);
  DrawQuad({false, blend}, full_screen_quad_vao_);

  // -----------------------------------------------------------------------------------------------

  // 3. Compute the rayleigh and mie single scattering for the current wavelengths, store them in
  // delta_rayleigh_scattering_texture and delta_mie_scattering_texture, and accumulate the
  // resulting luminance via additive blending in multiple_scattering_texture_ and
  // single_mie_scattering_texture_. The rayleigh scattering is stored together with the multiple
  // scattering contributions in multiple_scattering_texture_.
  glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, delta_rayleigh_scattering_texture, 0);
  glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, delta_mie_scattering_texture, 0);
  glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT2, multiple_scattering_texture_, 0);
  glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT3, single_mie_scattering_texture_, 0);
  glDrawBuffers(4, kDrawBuffers);

  glViewport(0, 0, SCATTERING_TEXTURE_WIDTH, SCATTERING_TEXTURE_HEIGHT);
  glScissor(0, 0, SCATTERING_TEXTURE_WIDTH, SCATTERING_TEXTURE_HEIGHT);
  compute_single_scattering.Use();
  compute_single_scattering.BindMat3("luminance_from_radiance", luminance_from_radiance);
  compute_single_scattering.BindTexture2d("transmittance_texture", transmittance_texture_, 0);
  compute_single_scattering.BindTexture2d("density_texture", density_texture_, 1);
  for (unsigned int layer = 0; layer < SCATTERING_TEXTURE_DEPTH; ++layer) {
    compute_single_scattering.BindInt("layer", layer);
    DrawQuad({false, false, blend, blend}, full_screen_quad_vao_);
  }

  // -----------------------------------------------------------------------------------------------

  // 4. Compute the 2nd, 3rd and 4th order of scattering, in sequence.
  for (unsigned int scattering_order = 2; scattering_order <= num_scattering_orders;
       ++scattering_order) {
    // 4.1. Compute the scattering density, and store it in delta_scattering_density_texture.
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, delta_scattering_density_texture, 0);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, 0, 0);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT2, 0, 0);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT3, 0, 0);
    glDrawBuffer(GL_COLOR_ATTACHMENT0);
    glViewport(0, 0, SCATTERING_TEXTURE_WIDTH, SCATTERING_TEXTURE_HEIGHT);
    glScissor(0, 0, SCATTERING_TEXTURE_WIDTH, SCATTERING_TEXTURE_HEIGHT);
    compute_scattering_density.Use();
    compute_scattering_density.BindTexture2d("phase_texture", phase_texture_, 0);
    compute_scattering_density.BindTexture2d("transmittance_texture", transmittance_texture_, 1);
    compute_scattering_density.BindTexture2d("density_texture", density_texture_, 2);
    compute_scattering_density.BindTexture3d(
        "single_rayleigh_scattering_texture", delta_rayleigh_scattering_texture, 3);
    compute_scattering_density.BindTexture3d(
        "single_mie_scattering_texture", delta_mie_scattering_texture, 4);
    compute_scattering_density.BindTexture3d(
        "multiple_scattering_texture", delta_multiple_scattering_texture, 5);
    compute_scattering_density.BindTexture2d("irradiance_texture", delta_irradiance_texture, 6);
    compute_scattering_density.BindInt("scattering_order", scattering_order);
    for (unsigned int layer = 0; layer < SCATTERING_TEXTURE_DEPTH; ++layer) {
      compute_scattering_density.BindInt("layer", layer);
      DrawQuad({false}, full_screen_quad_vao_);
    }

    // 4.2. Compute the indirect irradiance, store it in delta_irradiance_texture and
    // accumulate it in irradiance_texture_.
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, delta_irradiance_texture, 0);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, irradiance_texture_, 0);
    glDrawBuffers(2, kDrawBuffers);
    glViewport(0, 0, IRRADIANCE_TEXTURE_WIDTH, IRRADIANCE_TEXTURE_HEIGHT);
    glScissor(0, 0, IRRADIANCE_TEXTURE_WIDTH, IRRADIANCE_TEXTURE_HEIGHT);
    compute_indirect_irradiance.Use();
    compute_indirect_irradiance.BindMat3("luminance_from_radiance", luminance_from_radiance);
    compute_indirect_irradiance.BindTexture2d("phase_texture", phase_texture_, 0);
    compute_indirect_irradiance.BindTexture3d(
        "single_rayleigh_scattering_texture", delta_rayleigh_scattering_texture, 1);
    compute_indirect_irradiance.BindTexture3d(
        "single_mie_scattering_texture", delta_mie_scattering_texture, 2);
    compute_indirect_irradiance.BindTexture3d(
        "multiple_scattering_texture", delta_multiple_scattering_texture, 3);
    compute_indirect_irradiance.BindInt("scattering_order", scattering_order - 1);
    DrawQuad({false, true}, full_screen_quad_vao_);

    // 4.3. Compute the multiple scattering, store it in delta_multiple_scattering_texture, and
    // accumulate it in multiple_scattering_texture_.
    glFramebufferTexture(
        GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, delta_multiple_scattering_texture, 0);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, multiple_scattering_texture_, 0);
    glDrawBuffers(2, kDrawBuffers);
    glViewport(0, 0, SCATTERING_TEXTURE_WIDTH, SCATTERING_TEXTURE_HEIGHT);
    glScissor(0, 0, SCATTERING_TEXTURE_WIDTH, SCATTERING_TEXTURE_HEIGHT);
    compute_multiple_scattering.Use();
    compute_multiple_scattering.BindMat3("luminance_from_radiance", luminance_from_radiance);
    compute_multiple_scattering.BindTexture2d("phase_texture", phase_texture_, 0);
    compute_multiple_scattering.BindTexture2d("transmittance_texture", transmittance_texture_, 1);
    compute_multiple_scattering.BindTexture3d(
        "scattering_density_texture", delta_scattering_density_texture, 2);
    for (unsigned int layer = 0; layer < SCATTERING_TEXTURE_DEPTH; ++layer) {
      compute_multiple_scattering.BindInt("layer", layer);
      DrawQuad({false, true}, full_screen_quad_vao_);
    }
  }

  glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, 0, 0);
  glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT2, 0, 0);
  glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT3, 0, 0);

  glFlush();
}

void Model::UpdatePhaseFunctionTexture(
    std::vector<ScatteringAtmosphereComponent> const& scatteringComponents,
    const Model::vec3&                                lambdas) {

  if (phase_texture_ != 0) {
    glDeleteTextures(1, &phase_texture_);
  }

  size_t numAngles = scatteringComponents.front().phase.size();

  std::vector<float> data;
  data.reserve(4 * scatteringComponents.size() * numAngles);

  for (size_t i(0); i < scatteringComponents.size(); ++i) {
    for (auto const& spectrum : scatteringComponents[i].phase) {
      data.push_back(Interpolate(wavelengths_, spectrum, lambdas[0]));
      data.push_back(Interpolate(wavelengths_, spectrum, lambdas[1]));
      data.push_back(Interpolate(wavelengths_, spectrum, lambdas[2]));
      data.push_back(0.f);
    }
  }

  phase_texture_ = NewTexture2d(
      numAngles, scatteringComponents.size(), GL_RGBA32F, GL_RGBA, GL_FLOAT, data.data());
}

} // namespace csp::atmospheres::models::schneegans::internal
