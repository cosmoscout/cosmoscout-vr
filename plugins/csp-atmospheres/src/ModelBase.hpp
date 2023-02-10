////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSP_ATMOSPHERES_MODEL_BASE_HPP
#define CSP_ATMOSPHERES_MODEL_BASE_HPP

#include <GL/glew.h>
#include <nlohmann/json.hpp>

namespace csp::atmospheres {

/// Any new atmospheric model must be derived from this class.
class ModelBase {
 public:
  /// Whenever the model parameters are changed, this method will be called. It should return true
  /// if the shader needed to be recompiled in response to this change.
  virtual bool init(nlohmann::json modelSettings, double planetRadius, double atmosphereRadius) = 0;

  /// Returns a fragment shader which you can link to your shader program. See the class description
  /// above for more details. You have to call init() befor accessing the shader.
  virtual GLuint getShader() const = 0;

  /// This will be called each time the atmosphere is drawn. Use this to set model-specific
  /// uniforms. Any required textures can be bound starting at the given texture unit. The method
  /// should return the next free texture unit. So if the atmospheric model does not require any
  /// textures, it should simply return startTextureUnit.
  virtual GLuint setUniforms(GLuint program, GLuint startTextureUnit) const = 0;
};

} // namespace csp::atmospheres

#endif // CSP_ATMOSPHERES_MODEL_BASE_HPP
