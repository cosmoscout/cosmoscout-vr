////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSP_ATMOSPHERES_UTILS_HPP
#define CSP_ATMOSPHERES_UTILS_HPP

#include <GL/glew.h>
#include <string>
#include <tuple>

#include <glm/glm.hpp>

namespace csp::atmospheres::utils {

// Loads a 2D tiff texture containing precomputed values for the atmosphere. This is for instance
// used for the transmittance texture. It returns a tuple containing the OpenGL texture handle and
// size of the texture.
std::tuple<GLuint, glm::ivec2> read2DTexture(std::string const& path);

// Loads a 3D tiff texture containing precomputed values for the atmosphere. This is for instance
// used for the single scattering texture. It returns a tuple containing the OpenGL texture handle
// and size of the texture.
std::tuple<GLuint, glm::ivec3> read3DTexture(std::string const& path);
} // namespace csp::atmospheres::utils

#endif // CSP_ATMOSPHERES_UTILS_HPP
