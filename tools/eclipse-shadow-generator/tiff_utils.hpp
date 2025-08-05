////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef TIFF_HPP
#define TIFF_HPP

#include <cstdint>
#include <string>
#include <vector>

namespace tiff_utils {
struct RGBATexture {
  uint32_t           width  = 0;
  uint32_t           height = 0;
  std::vector<float> data;
};

// This helper function returns the number of layers in the given tiff file.
uint32_t getNumLayers(std::string const& path);

// This helper function reads a 2D RGB tiff image from the given path and returns the data as a
// vector of floats. The data is stored in the order R, G, B, A, R, G, B, A, ... The alpha channel
// is always 1.0 but is still included in the returned vector for easier upload to the CUDA device.
// The optional layer parameter can be used to read a specific layer from a multi-layer tiff file.
RGBATexture read2DTexture(std::string const& path, uint32_t layer = 0);

void write2D(std::string const& path, float* texture, int width, int height, int components);
void write3D(
    std::string const& path, float* texture, int width, int height, int depth, int components);

} // namespace tiff_utils

#endif // TIFF_HPP