////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSP_ATMOSPHERES_MODELS_COSMOSCOUT_MODEL_HPP
#define CSP_ATMOSPHERES_MODELS_COSMOSCOUT_MODEL_HPP

#include "../../../../src/cs-core/Settings.hpp"
#include "../../ModelBase.hpp"

#include <VistaOGLExt/VistaGLSLShader.h>

namespace csp::atmospheres::models::cosmoscout {

/// This atmospheric model uses a pretty basic implementation of single scattering. It requires no
/// preprocessing.
class Model : public ModelBase {
 public:
  /// The model parameters can be configured via the settings. An example parametrization is
  /// given in README.md. The default values below are used if parsing the settings failed.
  struct Settings {
    float     mMieHeight          = 1200.0; ///< In meters.
    glm::vec3 mMieScattering      = glm::vec3(4.0e-5);
    float     mMieAnisotropy      = 0.76;
    float     mRayleighHeight     = 8000.0; ///< In meters.
    glm::vec3 mRayleighScattering = glm::vec3(5.1768e-6, 12.2588e-6, 30.5964e-6);
    float     mRayleighAnisotropy = 0.0;

    /// Increasing those will improve the quality at the cost of a higher performance impact.
    cs::utils::DefaultProperty<int> mPrimaryRaySteps{7};
    cs::utils::DefaultProperty<int> mSecondaryRaySteps{3};
  };

  /// Whenever the model parameters are changed, this method needs to be called. It will return true
  /// if the shader needed to be recompiled. If that's the case, you can retrieve the new shader
  /// with the getShader() method below.
  bool init(
      nlohmann::json const& modelSettings, double planetRadius, double atmosphereRadius) override;

  /// Returns a fragment shader which you can link to your shader program. See the ModelBase class
  /// for more details. You have to call init() for accessing the shader.
  GLuint getShader() const override;

  /// This model sets no texture uniforms. So it will simply return startTextureUnit.
  GLuint setUniforms(GLuint program, GLuint startTextureUnit) const override;

 private:
  VistaGLSLShader mShader;
};

} // namespace csp::atmospheres::models::cosmoscout

#endif // CSP_ATMOSPHERES_MODELS_COSMOSCOUT_MODEL_HPP
