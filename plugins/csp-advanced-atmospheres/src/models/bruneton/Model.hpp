////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSP_ATMOSPHERES_MODELS_BRUNETON_MODEL_HPP
#define CSP_ATMOSPHERES_MODELS_BRUNETON_MODEL_HPP

#include "../../ModelBase.hpp"
#include "internal/Model.hpp"

namespace csp::atmospheres::models::bruneton {

class Model : public ModelBase {
 public:
  struct Settings {};

  bool init(nlohmann::json modelSettings, double planetRadius) override;

  GLuint getShader() const override;
  GLuint setUniforms(GLuint program, GLuint startTextureUnit) const override;

 private:
  Settings                         mSettings;
  nlohmann::json                   mPreviousSettings;
  double                           mPlanetRadius;
  std::unique_ptr<internal::Model> mModel;
};

} // namespace csp::atmospheres::models::bruneton

#endif // CSP_ATMOSPHERES_MODELS_BRUNETON_MODEL_HPP
