////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CS_GRAPHICS_STB_IMAGE_HELPER
#define CS_GRAPHICS_STB_IMAGE_HELPER

#include <GL/glew.h>
#include <stb_image.h>

namespace cs::graphics::internal {

/// Maps STBI image formats to OpenGL image formats.
// constexpr GLenum stbi_component_to_format(int component)
inline GLenum stbi_component_to_format(int component) {
  switch (component) {
  case STBI_grey:
    return GL_RED;
  case STBI_grey_alpha:
    return GL_RG;
  case STBI_rgb:
    return GL_RGB;
  case STBI_rgb_alpha:
    return GL_RGBA;
  default:
    return GL_RGB;
  }
}

/// Maps STBI image formats to OpenGL image formats.
// constexpr GLint stbi_component_to_internal_format(int component)
inline GLint stbi_component_to_internal_format(int component) {
  switch (component) {
  case STBI_grey:
    return GL_RED;
  case STBI_grey_alpha:
    return GL_RG;
  case STBI_rgb:
    return GL_RGB;
  case STBI_rgb_alpha:
    return GL_RGBA;
  default:
    return GL_RGB;
  }
}
} // namespace cs::graphics::internal

#endif // CS_GRAPHICS_STB_IMAGE_HELPER
