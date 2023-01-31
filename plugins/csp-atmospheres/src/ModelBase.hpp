////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSP_ATMOSPHERES_MODEL_BASE_HPP
#define CSP_ATMOSPHERES_MODEL_BASE_HPP

#include "../../../src/cs-utils/Property.hpp"

#include <GL/glew.h>
#include <glm/glm.hpp>
#include <nlohmann/json.hpp>

namespace csp::atmospheres {

class ModelBase {
 public:
  virtual ~ModelBase() = default;

  /// Whenever the model parameters are changed, this method will be called. It should return true
  /// if the shader needed to be recompiled.
  virtual bool init(nlohmann::json modelSettings, double planetRadius, double atmosphereRadius) = 0;

  /// Returns a fragment shader which you can link to your shader program. See the ModelBase class
  /// for more details. You have to call init() for accessing the shader.
  virtual GLuint getShader() const = 0;

  /// This model sets no texture uniforms. So it will simply return startTextureUnit.
  virtual GLuint setUniforms(GLuint program, GLuint startTextureUnit) const = 0;
};

} // namespace csp::atmospheres

#endif // CSP_ATMOSPHERES_MODEL_BASE_HPP
