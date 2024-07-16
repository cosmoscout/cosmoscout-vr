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
struct RGBTexture {
  uint32_t           width  = 0;
  uint32_t           height = 0;
  uint32_t           depth  = 0;
  std::vector<float> data;
};

RGBTexture read2DTexture(std::string const& path);
RGBTexture read3DTexture(std::string const& path);

} // namespace tiff_utils

#endif // TIFF_HPP