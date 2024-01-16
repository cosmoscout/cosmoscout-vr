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

#ifndef CSP_ATMOSPHERES_MODELS_BRUNETON_INTERNAL_MODEL_HPP
#define CSP_ATMOSPHERES_MODELS_BRUNETON_INTERNAL_MODEL_HPP

#include <GL/glew.h>
#include <array>
#include <functional>
#include <glm/glm.hpp>
#include <string>
#include <vector>

namespace csp::atmospheres::models::bruneton::internal {

struct ScatteringAtmosphereComponent {
  // The outer vector contains entries for each angle of the phase function. The first item
  // corresponds to 0° (forward scattering), the last item to 180° (back scattering). The inner
  // vectors contain the intensity values for each wavelength at the specific angle.
  std::vector<std::vector<double>> mPhase;

  // Beta_sca per wavelength for the altitude where density is 1.0.
  std::vector<double> mScattering;

  // Beta_abs per wavelength for the altitude where density is 1.0.
  std::vector<double> mAbsorption;

  // Linear function describing the density distribution from bottom to top. The value at a specific
  // altitude will be multiplied with the Beta_sca and Beta_abs values above.
  std::vector<double> mDensity;
};

struct AbsorbingAtmosphereComponent {
  // Beta_abs per wavelength for N_0
  std::vector<double> mAbsorption;

  // Linear function describing the density distribution from bottom to top. The value at a specific
  // altitude will be multiplied with the Beta_sca and Beta_abs values above.
  std::vector<double> mDensity;
};

struct ModelParams {
  // The wavelength values, in nanometers, and sorted in increasing order, for/ which the
  // solar_irradiance, molecules_scattering, aerosols_scattering, aerosols_extinction and
  // ground_albedo samples are provided. If your shaders use luminance values (as opposed to
  // radiance values, see above), use a large number of wavelengths (e.g. between 15 and 50) to get
  // accurate results (this number of wavelengths has absolutely no impact on the shader
  // performance).
  std::vector<double> mWavelengths;

  ScatteringAtmosphereComponent mMolecules;

  ScatteringAtmosphereComponent mAerosols;

  AbsorbingAtmosphereComponent mOzone;

  // The sun's angular radius, in radians. Warning: the implementation uses approximations that are
  // valid only if this value is smaller than 0.1.
  double mSunAngularRadius;

  // The distance between the planet center and the bottom of the atmosphere, in m.
  double mBottomRadius;

  // The distance between the planet center and the top of the atmosphere, in m.
  double mTopRadius;

  // The average albedo of the ground.
  double mGroundAlbedo;

  // The maximum Sun zenith angle for which atmospheric scattering must be precomputed, in radians
  // (for maximum precision, use the smallest Sun zenith angle yielding negligible sky light
  // radiance values. For instance, for the Earth case, 102 degrees is a good choice for most cases
  // (120 degrees is necessary for very high exposure values).
  double mMaxSunZenithAngle;

  int32_t mSampleCountOpticalDepth;
  int32_t mSampleCountSingleScattering;
  int32_t mSampleCountMultiScattering;
  int32_t mSampleCountScatteringDensity;
  int32_t mSampleCountIndirectIrradiance;
  int32_t mTransmittanceTextureWidth;
  int32_t mTransmittanceTextureHeight;
  int32_t mScatteringTextureRSize;
  int32_t mScatteringTextureMuSize;
  int32_t mScatteringTextureMuSSize;
  int32_t mScatteringTextureNuSize;
  int32_t mIrradianceTextureWidth;
  int32_t mIrradianceTextureHeight;
};

class Model {
 public:
  Model(ModelParams params);

  ~Model();

  void Init(unsigned int num_scattering_orders = 4);

  GLuint shader() const {
    return atmosphere_shader_;
  }

  void SetProgramUniforms(GLuint program, GLuint phase_texture_unit,
      GLuint transmittance_texture_unit, GLuint multiple_scattering_texture_unit,
      GLuint irradiance_texture_unit, GLuint single_aerosols_scattering_texture_unit) const;

  static constexpr double kLambdaR = 680.0;
  static constexpr double kLambdaG = 550.0;
  static constexpr double kLambdaB = 440.0;

 private:
  void Precompute(GLuint fbo, GLuint delta_irradiance_texture,
      GLuint delta_molecules_scattering_texture, GLuint delta_aerosols_scattering_texture,
      GLuint delta_scattering_density_texture, GLuint delta_multiple_scattering_texture,
      const glm::dvec3& lambdas, const glm::mat3& luminance_from_radiance, bool blend,
      unsigned int num_scattering_orders);

  void UpdatePhaseFunctionTexture(
      std::vector<ScatteringAtmosphereComponent> const& scatteringComponents,
      const glm::dvec3&                                 lambdas);

  const ModelParams params_;
  const int32_t     mScatteringTextureWidth;
  const int32_t     mScatteringTextureHeight;
  const int32_t     mScatteringTextureDepth;

  std::function<std::string(const glm::dvec3&)> glsl_header_factory_;
  GLuint                                        phase_texture_         = 0;
  GLuint                                        density_texture_       = 0;
  GLuint                                        transmittance_texture_ = 0;

  // This texture stores single molecules scattering plus all multiple scattering contributions. The
  // single aerosols scattering is stored in an extra texture to have a higher angular resolution
  // (the phase function is applied at render time).
  GLuint multiple_scattering_texture_        = 0;
  GLuint single_aerosols_scattering_texture_ = 0;

  GLuint irradiance_texture_   = 0;
  GLuint atmosphere_shader_    = 0;
  GLuint full_screen_quad_vao_ = 0;
  GLuint full_screen_quad_vbo_ = 0;
};

} // namespace csp::atmospheres::models::bruneton::internal

#endif // CSP_ATMOSPHERES_MODELS_BRUNETON_INTERNAL_MODEL_HPP
