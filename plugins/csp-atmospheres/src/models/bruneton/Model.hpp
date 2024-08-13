////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSP_ATMOSPHERES_MODELS_BRUNETON_MODEL_HPP
#define CSP_ATMOSPHERES_MODELS_BRUNETON_MODEL_HPP

#include "../../../../src/cs-core/Settings.hpp"
#include "../../ModelBase.hpp"

namespace csp::atmospheres::models::bruneton {

/// This atmospheric model is based on an implementation of multiple-scattering by Eric Bruneton.
/// The main difference to the original implementation is that this variant uses phase functions,
/// extinction coefficients, and density distributions loaded from CSV files instead of analytic
/// descriptions.
/// Besides, we refactored out the precomputation step into a separate executable, which is
/// responsible for generating the textures needed for rendering. This way, we can increase the
/// fidelity of the preprocessing step without affecting the startup time of the application.
/// More information on the original implementation can be found in the repo by Eric Bruneton:
/// https://github.com/ebruneton/precomputed_atmospheric_scattering as well as in his paper
/// "Precomputed Atmospheric Scattering" (https://hal.inria.fr/inria-00288758/en).
class Model : public ModelBase {
 public:
  /// The settings of this model are extremely simple. They only contain the path to the directory
  /// where the precomputed textures are stored.
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

  GLuint mPhaseTexture          = 0;
  GLuint mTransmittanceTexture  = 0;
  GLuint mThetaDeviationTexture = 0;
  GLuint mIrradianceTexture     = 0;

  GLuint mAtmosphereShader = 0;
};

} // namespace csp::atmospheres::models::bruneton

#endif // CSP_ATMOSPHERES_MODELS_BRUNETON_MODEL_HPP
