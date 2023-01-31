////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSP_ATMOSPHERES_MODELS_BRUNETON_MODEL_HPP
#define CSP_ATMOSPHERES_MODELS_BRUNETON_MODEL_HPP

#include "../../../../src/cs-core/Settings.hpp"
#include "../../ModelBase.hpp"
#include "internal/Model.hpp"

namespace csp::atmospheres::models::bruneton {

/// This atmospheric model uses an implementation of multi-scattering by Eric Bruneton. More
/// information can be found in the repo
/// https://github.com/ebruneton/precomputed_atmospheric_scattering as well as in the paper
/// "Precomputed Atmospheric Scattering" (https://hal.inria.fr/inria-00288758/en).
class Model : public ModelBase {
 public:
  /// Some of the model parameters can be configured via the settings. An example parametrization is
  /// given in README.md, more details can be found in the paper "Precomputed Atmospheric
  /// Scattering" by Eric Bruneton.
  struct Settings {
    double mSunAngularRadius{};
    double mRayleigh{};
    double mRayleighScaleHeight{}; ///< In meters.
    double mMieScaleHeight{};      ///< In meters.
    double mMieAngstromAlpha{};
    double mMieAngstromBeta{};
    double mMieSingleScatteringAlbedo{};
    double mMiePhaseFunctionG{};

    cs::utils::DefaultProperty<double> mGroundAlbedo{0.1};
    cs::utils::DefaultProperty<bool>   mUseOzone{false};
  };

  /// Whenever the model parameters are changed, this method needs to be called. It will return true
  /// if the shader needed to be recompiled. If that's the case, you can retrieve the new shader
  /// with the getShader() method below.
  bool init(nlohmann::json modelSettings, double planetRadius, double atmosphereRadius) override;

  /// Returns a fragment shader which you can link to your shader program. See the ModelBase class
  /// for more details. You have to call init() for accessing the shader.
  GLuint getShader() const override;

  /// This model sets three texture uniforms. So it will return startTextureUnit + 3.
  GLuint setUniforms(GLuint program, GLuint startTextureUnit) const override;

 private:
  Settings                         mSettings;
  nlohmann::json                   mPreviousSettings;
  double                           mPlanetRadius;
  double                           mAtmosphereRadius;
  std::unique_ptr<internal::Model> mModel;
};

} // namespace csp::atmospheres::models::bruneton

#endif // CSP_ATMOSPHERES_MODELS_BRUNETON_MODEL_HPP