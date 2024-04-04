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

    std::string mPhaseTexture;
    std::string mTransmittanceTexture;
    std::string mIrradianceTexture;
    std::string mSingleScatteringTexture;
    std::string mMultipleScatteringTexture;

    /// The angular radius of the Sun needs to be specified. As SPICE is not fully available when
    /// the plugin is loaded, we cannot compute it. Also, this actually varies in reality.
    float     mSunAngularRadius = 0.004675F;
    glm::vec3 mSunIlluminance   = glm::vec3(144810, 129444, 127099);

    /// The resolution of the transmittance texture. Larger values can improve the sampling of thin
    /// atmospheric layers close to the horizon.
    cs::utils::DefaultProperty<int32_t> mTransmittanceTextureWidth{256};
    cs::utils::DefaultProperty<int32_t> mTransmittanceTextureHeight{64};

    /// Larger values improve sampling of thick low-altitude layers.
    cs::utils::DefaultProperty<int32_t> mScatteringTextureRSize{32};

    /// Larger values reduce circular banding artifacts around zenith for thick atmospheres.
    cs::utils::DefaultProperty<int32_t> mScatteringTextureMuSize{128};

    /// Larger values reduce banding in the day-night transition when seen from space.
    cs::utils::DefaultProperty<int32_t> mScatteringTextureMuSSize{32};

    /// Larger values reduce circular banding artifacts around sun for thick atmospheres.
    cs::utils::DefaultProperty<int32_t> mScatteringTextureNuSize{8};

    /// The resolution of the irradiance texture.
    cs::utils::DefaultProperty<int32_t> mIrradianceTextureWidth{64};
    cs::utils::DefaultProperty<int32_t> mIrradianceTextureHeight{16};

    /// The maximum Sun zenith angle for which atmospheric scattering must be precomputed, in
    /// radians (for maximum precision, use the smallest Sun zenith angle yielding negligible sky
    /// light radiance values. For instance, for the Earth case, 102 degrees is a good choice for
    /// most cases (120 degrees is necessary for very high exposure values).
    cs::utils::DefaultProperty<float> mMaxSunZenithAngle{120.F / 180.F * glm::pi<float>()};
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
