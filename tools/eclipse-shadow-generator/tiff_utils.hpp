////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef TIFF_HPP
#define TIFF_HPP

#include <cstdint>
#include <string>
#include <tuple>
#include <vector>

namespace tiff_utils {
struct RGBATexture {
  uint32_t           width  = 0;
  uint32_t           height = 0;
  std::vector<float> data;
};

RGBATexture read2DTexture(std::string const& path, uint32_t layer = 0);

} // namespace tiff_utils

#endif // TIFF_HPP