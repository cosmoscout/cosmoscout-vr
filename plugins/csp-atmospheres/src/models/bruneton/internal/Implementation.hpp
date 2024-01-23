////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-FileCopyrightText: 2017 Eric Bruneton
// SPDX-License-Identifier: BSD-3-Clause

#ifndef CSP_ATMOSPHERES_MODELS_BRUNETON_INTERNAL_MODEL_HPP
#define CSP_ATMOSPHERES_MODELS_BRUNETON_INTERNAL_MODEL_HPP

#include <GL/glew.h>
#include <array>
#include <functional>
#include <glm/glm.hpp>
#include <string>
#include <vector>

/// The C++ implementation of this atmospheric model is based on this class by Eric Bruneton:
/// https://github.com/ebruneton/precomputed_atmospheric_scattering/blob/master/atmosphere/model.h
/// While we refactored / restyled large parts of the code, the overall flow of control remains the
/// same.

namespace csp::atmospheres::models::bruneton::internal {

/// These parameters are passed to the atmospheric scattering model.
struct Params {

  /// This struct basically corresponds to the Model::Settings::ScatteringComponent struct, however
  /// it contains the actual values loaded from the CSV files.
  struct ScatteringComponent {
    /// The outer vector contains entries for each angle of the phase function. The first item
    /// corresponds to 0° (forward scattering), the last item to 180° (back scattering). The inner
    /// vectors contain the intensity values for each wavelength at the specific angle.
    std::vector<std::vector<float>> mPhase;

    /// Beta_sca per wavelength for the altitude where density is 1.0.
    std::vector<float> mScattering;

    /// Beta_abs per wavelength for the altitude where density is 1.0.
    std::vector<float> mAbsorption;

    /// Linear function describing the density distribution from bottom to top. The value at a
    /// specific altitude will be multiplied with the Beta_sca and Beta_abs values above.
    std::vector<float> mDensity;
  };

  /// This struct basically corresponds to the Model::Settings::AbsorbingComponent struct, however
  /// it contains the actual values loaded from the CSV files.
  struct AbsorbingComponent {
    /// Beta_abs per wavelength for N_0
    std::vector<float> mAbsorption;

    /// Linear function describing the density distribution from bottom to top. The value at a
    /// specific altitude will be multiplied with the Beta_sca and Beta_abs values above.
    std::vector<float> mDensity;
  };

  /// The atmosphere can contain these components. If no absorbing component was configured, all
  /// mAbsorption of mOzone will be zero.
  ScatteringComponent mMolecules;
  ScatteringComponent mAerosols;
  AbsorbingComponent  mOzone;

  /// The wavelength values, in nanometers, and sorted in increasing order, for which the
  /// phase functions and extinction coefficients in the atmosphere components are given.
  std::vector<float> mWavelengths;

  /// See the Model class header for an explanation of these properties.
  float  mSunAngularRadius;
  float  mBottomRadius;
  float  mTopRadius;
  float  mGroundAlbedo;
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

  /// The maximum Sun zenith angle for which atmospheric scattering must be precomputed, in radians
  /// (for maximum precision, use the smallest Sun zenith angle yielding negligible sky light
  /// radiance values. For instance, for the Earth case, 102 degrees is a good choice for most cases
  /// (120 degrees is necessary for very high exposure values).
  float mMaxSunZenithAngle;
};

class Implementation {
 public:
  /// If only three wavelengths are used during rendering, these three are used:
  static constexpr float kLambdaR = 680.0;
  static constexpr float kLambdaG = 550.0;
  static constexpr float kLambdaB = 440.0;

  /// The constructor of the class takes all parameters which define the attributes of the
  /// atmosphere. It will allocate various GPU resources.
  Implementation(Params params);
  ~Implementation();

  /// This will preprocess the multiple scattering up to the given number. Setting this to one will
  /// disable multiple scattering.
  void init(unsigned int numScatteringOrders);

  /// Returns a fragment shader which you can link into a shader program.
  GLuint shader() const;

  /// Sets all required uniforms. The given shader program should have the shader linked into, which
  /// got returned by the method above.
  void setProgramUniforms(GLuint program, GLuint phaseTextureUnit, GLuint transmittanceTextureUnit,
      GLuint multipleScatteringTextureUnit, GLuint irradianceTextureUnit,
      GLuint singleAerosolsScatteringTextureUnit) const;

 private:
  void precompute(GLuint fbo, GLuint deltaIrradianceTexture, GLuint deltaMoleculesScatteringTexture,
      GLuint deltaAerosolsScatteringTexture, GLuint deltaScatteringDensityTexture,
      GLuint deltaMultipleScatteringTexture, glm::vec3 const& lambdas,
      glm::mat3 const& luminanceFromRadiance, bool blend, unsigned int numScatteringOrders);

  void updatePhaseFunctionTexture(
      std::vector<Params::ScatteringComponent> const& scatteringComponents,
      glm::vec3 const&                               lambdas);

  const Params  mParams;
  const int32_t mScatteringTextureWidth;
  const int32_t mScatteringTextureHeight;
  const int32_t mScatteringTextureDepth;

  std::function<std::string(glm::vec3 const&)> mGlslHeaderFactory;

  // To optimize resource usage, this texture stores single molecule-scattering plus all
  // multiple-scattering contributions. The single aerosols scattering is stored in an extra
  // texture.
  GLuint mMultipleScatteringTexture       = 0;
  GLuint mSingleAerosolsScatteringTexture = 0;

  GLuint mPhaseTexture         = 0;
  GLuint mDensityTexture       = 0;
  GLuint mTransmittanceTexture = 0;
  GLuint mIrradianceTexture    = 0;
  GLuint mAtmosphereShader     = 0;
  GLuint mFullScreenQuadVAO    = 0;
  GLuint mFullScreenQuadVBO    = 0;
};

} // namespace csp::atmospheres::models::bruneton::internal

#endif // CSP_ATMOSPHERES_MODELS_BRUNETON_INTERNAL_MODEL_HPP
