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

class Model : public ModelBase {
 public:
  struct Settings {
    float     mMieHeight{}; ///< In meters.
    glm::vec3 mMieScattering{};
    float     mMieAnisotropy{};
    float     mRayleighHeight{}; ///< In meters.
    glm::vec3 mRayleighScattering{};
    float     mRayleighAnisotropy{};

    cs::utils::DefaultProperty<int> mPrimaryRaySteps{7};
    cs::utils::DefaultProperty<int> mSecondaryRaySteps{3};
  };

  bool init(nlohmann::json modelSettings, double planetRadius, double atmosphereRadius) override;

  GLuint getShader() const override;
  GLuint setUniforms(GLuint program, GLuint startTextureUnit) const override;

 private:
  Settings        mSettings;
  nlohmann::json  mPreviousSettings;
  VistaGLSLShader mShader;
  double          mPlanetRadius;
  double          mAtmosphereRadius;
};

} // namespace csp::atmospheres::models::cosmoscout

#endif // CSP_ATMOSPHERES_MODELS_COSMOSCOUT_MODEL_HPP
