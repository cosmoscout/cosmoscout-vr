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

  virtual bool init(nlohmann::json modelSettings, double planetRadius, double atmosphereRadius) = 0;

  virtual GLuint getShader() const = 0;

  virtual GLuint setUniforms(GLuint program, GLuint startTextureUnit) const = 0;
};

} // namespace csp::atmospheres

#endif // CSP_ATMOSPHERES_MODEL_BASE_HPP
