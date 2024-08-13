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

namespace csp::atmospheres::utils {
std::tuple<GLuint, int32_t, int32_t>          read2DTexture(std::string const& path);
std::tuple<GLuint, int32_t, int32_t, int32_t> read3DTexture(std::string const& path);
} // namespace csp::atmospheres::utils

#endif // CSP_ATMOSPHERES_UTILS_HPP
