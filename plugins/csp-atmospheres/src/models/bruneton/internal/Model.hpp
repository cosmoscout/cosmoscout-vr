////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-FileCopyrightText: 2017 Eric Bruneton
// SPDX-License-Identifier: BSD-3-Clause

// This file has been directly copied from here:
// https://github.com/ebruneton/precomputed_atmospheric_scattering/blob/master/atmosphere/model.h
// The documentation below can also be read online at:
// https://ebruneton.github.io/precomputed_atmospheric_scattering/atmosphere/model.h.html
// Changes to this file are mostly related to formatting. The only other change with respect to the
// original code is the removal of the "normal" parameter from the documentation of the
// GetSunAndSkyIrradiance() and GetSunAndSkyIlluminance() methods. In the original implementation,
// these methods used to premultiply the irradiance with the dot product between light direction and
// surface normal. As this factor is already included in the BRDFs used in CosmoCout VR, we have
// removed this and adapted the documentation here accordingly.
// Similarily, the shadow_length parameter has been removed from the public API as this is currently
// not supported by CosmoScout VR.

/*<h2>atmosphere/model.h</h2>

<p>This file defines the API to use our atmosphere model in OpenGL applications.
To use it:
<ul>
<li>create a <code>Model</code> instance with the desired atmosphere
parameters.</li>
<li>call <code>Init</code> to precompute the atmosphere textures,</li>
<li>link <code>GetShader</code> with your shaders that need access to the
atmosphere shading functions.</li>
<li>for each GLSL program linked with <code>GetShader</code>, call
<code>SetProgramUniforms</code> to bind the precomputed textures to this
program (usually at each frame).</li>
<li>delete your <code>Model</code> when you no longer need its shader and
precomputed textures (the destructor deletes these resources).</li>
</ul>

<p>The shader returned by <code>GetShader</code> provides the following
functions (that you need to forward declare in your own shaders to be able to
compile them separately):

<pre class="prettyprint">
// Returns the radiance of the Sun, outside the atmosphere.
vec3 GetSolarRadiance();

// Returns the sky radiance along the segment from 'camera' to the nearest
// atmosphere boundary in direction 'view_ray', as well as the transmittance
// along this segment.
vec3 GetSkyRadiance(vec3 camera, vec3 view_ray,
    vec3 sun_direction, out vec3 transmittance);

// Returns the sky radiance along the segment from 'camera' to 'p', as well as
// the transmittance along this segment.
vec3 GetSkyRadianceToPoint(vec3 camera, vec3 p,
    vec3 sun_direction, out vec3 transmittance);

// Returns the sun and sky irradiance received on a surface patch located at 'p'.
vec3 GetSunAndSkyIrradiance(vec3 p, vec3 sun_direction, out vec3 sky_irradiance);

// Returns the luminance of the Sun, outside the atmosphere.
vec3 GetSolarLuminance();

// Returns the sky luminance along the segment from 'camera' to the nearest
// atmosphere boundary in direction 'view_ray', as well as the transmittance
// along this segment.
vec3 GetSkyLuminance(vec3 camera, vec3 view_ray,
    vec3 sun_direction, out vec3 transmittance);

// Returns the sky luminance along the segment from 'camera' to 'p', as well as
// the transmittance along this segment.
vec3 GetSkyLuminanceToPoint(vec3 camera, vec3 p,
    vec3 sun_direction, out vec3 transmittance);

// Returns the sun and sky illuminance received on a surface patch located at 'p'.
vec3 GetSunAndSkyIlluminance(vec3 p, vec3 sun_direction, out vec3 sky_illuminance);
</pre>

<p>where
<ul>
<li><code>camera</code> and <code>p</code> must be expressed in a reference
frame where the planet center is at the origin, and measured in the unit passed
to the constructor's <code>length_unit_in_meters</code> argument.
<code>camera</code> can be in space, but <code>p</code> must be inside the
atmosphere,</li>
<li><code>view_ray</code>, <code>sun_direction</code> and <code>normal</code>
are unit direction vectors expressed in the same reference frame (with
<code>sun_direction</code> pointing <i>towards</i> the Sun),</li>
<li><code>shadow_length</code> is the length along the segment which is in
shadow, measured in the unit passed to the constructor's
<code>length_unit_in_meters</code> argument.</li>
</ul>

<p>and where
<ul>
<li>the first 4 functions return spectral radiance and irradiance values
(in $W.m^{-2}.sr^{-1}.nm^{-1}$ and $W.m^{-2}.nm^{-1}$), at the 3 wavelengths
<code>kLambdaR</code>, <code>kLambdaG</code>, <code>kLambdaB</code> (in this
order),</li>
<li>the other functions return luminance and illuminance values (in
$cd.m^{-2}$ and $lx$) in linear <a href="https://en.wikipedia.org/wiki/SRGB">
sRGB</a> space (i.e. before adjustements for gamma correction),</li>
<li>all the functions return the (unitless) transmittance of the atmosphere
along the specified segment at the 3 wavelengths <code>kLambdaR</code>,
<code>kLambdaG</code>, <code>kLambdaB</code> (in this order).</li>
</ul>

<p><b>Note</b> The precomputed atmosphere textures can store either irradiance
or illuminance values (see the <code>num_precomputed_wavelengths</code>
parameter):
<ul>
  <li>when using irradiance values, the RGB channels of these textures contain
  spectral irradiance values, in $W.m^{-2}.nm^{-1}$, at the 3 wavelengths
  <code>kLambdaR</code>, <code>kLambdaG</code>, <code>kLambdaB</code> (in this
  order). The API functions returning radiance values return these precomputed
  values (times the phase functions), while the API functions returning
  luminance values use the approximation described in
  <a href="https://arxiv.org/pdf/1612.04336.pdf">A Qualitative and Quantitative
  Evaluation of 8 Clear Sky Models</a>, section 14.3, to convert 3 radiance
  values to linear sRGB luminance values.</li>
  <li>when using illuminance values, the RGB channels of these textures contain
  illuminance values, in $lx$, in linear sRGB space. These illuminance values
  are precomputed as described in
  <a href="http://www.oskee.wz.cz/stranka/uploads/SCCG10ElekKmoch.pdf">Real-time
  Spectral Scattering in Large-scale Natural Participating Media</a>, section
  4.4 (i.e. <code>num_precomputed_wavelengths</code> irradiance values are
  precomputed, and then converted to sRGB via a numerical integration of this
  spectrum with the CIE color matching functions). The API functions returning
  luminance values return these precomputed values (times the phase functions),
  while <i>the API functions returning radiance values are not provided</i>.
  </li>
</ul>

<p>The concrete API definition is the following:
*/

#ifndef CSP_ATMOSPHERES_MODELS_BRUNETON_INTERNAL_MODEL_HPP
#define CSP_ATMOSPHERES_MODELS_BRUNETON_INTERNAL_MODEL_HPP

#include <GL/glew.h>
#include <array>
#include <functional>
#include <string>
#include <vector>

namespace csp::atmospheres::models::bruneton::internal {

struct ScatteringAtmosphereComponent {
  // The outer vector contains entries for each angle of the phase function. The first item
  // corresponds to 0° (forward scattering), the last item to 180° (back scattering). The inner
  // vectors contain the intensity values for each wavelength at the specific angle.
  std::vector<std::vector<double>> phase;

  // Beta_sca per wavelength for N_0
  std::vector<double> scattering;

  // Beta_abs per wavelength for N_0
  std::vector<double> absorption;

  // Linear function describing the density distribution from bottom to top. The value at a specific
  // altitude will be multiplied with the Beta_sca and Beta_abs values above.
  std::vector<double> density;
};

struct AbsorbingAtmosphereComponent {
  // Beta_abs per wavelength for N_0
  std::vector<double> absorption;

  // Linear function describing the density distribution from bottom to top. The value at a specific
  // altitude will be multiplied with the Beta_sca and Beta_abs values above.
  std::vector<double> density;
};

class Model {
 public:
  Model(
      // The wavelength values, in nanometers, and sorted in increasing order, for
      // which the solar_irradiance, rayleigh_scattering, mie_scattering,
      // mie_extinction and ground_albedo samples are provided. If your shaders
      // use luminance values (as opposed to radiance values, see above), use a
      // large number of wavelengths (e.g. between 15 and 50) to get accurate
      // results (this number of wavelengths has absolutely no impact on the
      // shader performance).
      const std::vector<double>& wavelengths,
      // The sun's angular radius, in radians. Warning: the implementation uses
      // approximations that are valid only if this value is smaller than 0.1.
      double sun_angular_radius,
      // The distance between the planet center and the bottom of the atmosphere,
      // in m.
      double bottom_radius,
      // The distance between the planet center and the top of the atmosphere,
      // in m.
      double top_radius,

      const ScatteringAtmosphereComponent& rayleigh,

      const ScatteringAtmosphereComponent& mie,

      const AbsorbingAtmosphereComponent& ozone,

      // The average albedo of the ground.
      double ground_albedo,
      // The maximum Sun zenith angle for which atmospheric scattering must be
      // precomputed, in radians (for maximum precision, use the smallest Sun
      // zenith angle yielding negligible sky light radiance values. For instance,
      // for the Earth case, 102 degrees is a good choice for most cases (120
      // degrees is necessary for very high exposure values).
      double max_sun_zenith_angle,
      // The length unit used in your shaders and meshes. This is the length unit
      // which must be used when calling the atmosphere model shader functions.
      double length_unit_in_meters);

  ~Model();

  void Init(unsigned int num_scattering_orders = 4);

  GLuint shader() const {
    return atmosphere_shader_;
  }

  void SetProgramUniforms(GLuint program, GLuint phase_texture_unit,
      GLuint transmittance_texture_unit, GLuint multiple_scattering_texture_unit,
      GLuint irradiance_texture_unit, GLuint single_mie_scattering_texture_unit) const;

  static constexpr double kLambdaR = 680.0;
  static constexpr double kLambdaG = 550.0;
  static constexpr double kLambdaB = 440.0;

 private:
  typedef std::array<double, 3> vec3;
  typedef std::array<float, 9>  mat3;

  void Precompute(GLuint fbo, GLuint delta_irradiance_texture,
      GLuint delta_rayleigh_scattering_texture, GLuint delta_mie_scattering_texture,
      GLuint delta_scattering_density_texture, GLuint delta_multiple_scattering_texture,
      const vec3& lambdas, const mat3& luminance_from_radiance, bool blend,
      unsigned int num_scattering_orders);

  void UpdatePhaseFunctionTexture(
      std::vector<ScatteringAtmosphereComponent> const& scatteringComponents,
      const Model::vec3&                                lambdas);

  std::vector<double> wavelengths_;

  ScatteringAtmosphereComponent rayleigh_;
  ScatteringAtmosphereComponent mie_;
  AbsorbingAtmosphereComponent  ozone_;

  std::function<std::string(const vec3&)> glsl_header_factory_;
  GLuint                                  phase_texture_         = 0;
  GLuint                                  density_texture_       = 0;
  GLuint                                  transmittance_texture_ = 0;

  // This texture stores single rayleigh scattering plus all multiple scattering contributions. The
  // single mie scattering is stored in an extra texture to have a higher angular resolution (the
  // phase function is applied at render time).
  GLuint multiple_scattering_texture_   = 0;
  GLuint single_mie_scattering_texture_ = 0;

  GLuint irradiance_texture_   = 0;
  GLuint atmosphere_shader_    = 0;
  GLuint full_screen_quad_vao_ = 0;
  GLuint full_screen_quad_vbo_ = 0;
};

} // namespace csp::atmospheres::models::bruneton::internal

#endif // CSP_ATMOSPHERES_MODELS_BRUNETON_INTERNAL_MODEL_HPP
