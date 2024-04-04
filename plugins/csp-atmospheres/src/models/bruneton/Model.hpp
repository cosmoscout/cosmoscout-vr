////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-FileCopyrightText: 2017 Eric Bruneton
// SPDX-License-Identifier: BSD-3-Clause

#ifndef CSP_ATMOSPHERES_MODELS_BRUNETON_MODEL_HPP
#define CSP_ATMOSPHERES_MODELS_BRUNETON_MODEL_HPP

#include "../../../../src/cs-core/Settings.hpp"
#include "../../ModelBase.hpp"

namespace csp::atmospheres::models::bruneton {

/// This atmospheric model is based on an implementation of multiple-scattering by Eric Bruneton.
/// The main difference to the original implementation is that this variant uses phase functions,
/// extinction coefficients, and density distributions loaded from CSV files instead of analytic
/// descriptions.
/// More information on the original implementation can be found in the repo by Eric Bruneton:
/// https://github.com/ebruneton/precomputed_atmospheric_scattering as well as in his paper
/// "Precomputed Atmospheric Scattering" (https://hal.inria.fr/inria-00288758/en).
/// The default values for the model parameters further down this file are based on the parameters
/// from Eric Bruneton:
/// https://github.com/ebruneton/precomputed_atmospheric_scattering/blob/master/atmosphere/constants.h
class Model : public ModelBase {
 public:
  /// If only three wavelengths are used during rendering, these three are used:
  static constexpr float kLambdaR = 680.0;
  static constexpr float kLambdaG = 550.0;
  static constexpr float kLambdaB = 440.0;

  struct Settings {
    std::string mDataDirectory;
  };

  virtual ~Model();

  /// Whenever the model parameters are changed, this method needs to be called. It will return true
  /// if the shader needed to be recompiled. If that's the case, you can retrieve the new shader
  /// with the getShader() method below.
  bool init(
      nlohmann::json const& modelSettings, double planetRadius, double atmosphereRadius) override;

  /// Returns a fragment shader which you can link into your shader program. See the ModelBase class
  /// for more details. You have to call init() befor accessing the shader.
  GLuint getShader() const override;

  /// This model sets five texture uniforms. So it will return startTextureUnit + 5.
  GLuint setUniforms(GLuint program, GLuint startTextureUnit) const override;

 private:
  std::tuple<GLuint, int32_t, int32_t>          read2DTexture(std::string const& path) const;
  std::tuple<GLuint, int32_t, int32_t, int32_t> read3DTexture(std::string const& path) const;

  int32_t mTransmittanceTextureWidth{};
  int32_t mTransmittanceTextureHeight{};
  int32_t mIrradianceTextureWidth{};
  int32_t mIrradianceTextureHeight{};
  int32_t mScatteringTextureRSize{};
  int32_t mScatteringTextureMuSize{};
  int32_t mScatteringTextureMuSSize{};
  int32_t mScatteringTextureNuSize{};

  // To optimize resource usage, this texture stores single molecule-scattering plus all
  // multiple-scattering contributions. The single aerosols scattering is stored in an extra
  // texture.
  GLuint mMultipleScatteringTexture       = 0;
  GLuint mSingleAerosolsScatteringTexture = 0;

  GLuint mPhaseTexture         = 0;
  GLuint mTransmittanceTexture = 0;
  GLuint mIrradianceTexture    = 0;

  GLuint mAtmosphereShader = 0;
};

} // namespace csp::atmospheres::models::bruneton

#endif // CSP_ATMOSPHERES_MODELS_BRUNETON_MODEL_HPP
