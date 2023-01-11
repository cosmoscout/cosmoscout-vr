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

class Model : public ModelBase {
 public:
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

  bool init(nlohmann::json modelSettings, double planetRadius, double atmosphereRadius) override;

  GLuint getShader() const override;
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
