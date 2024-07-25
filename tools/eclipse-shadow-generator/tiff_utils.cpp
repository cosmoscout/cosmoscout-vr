////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "tiff_utils.hpp"

#include <iostream>
#include <tiffio.h>

namespace {

////////////////////////////////////////////////////////////////////////////////////////////////////

std::vector<float> addAlphaChannel(std::vector<float> const& rgbData) {
  std::vector<float> rgbaData(rgbData.size() * 4 / 3);

  for (unsigned i = 0; i < rgbData.size() / 3; i++) {
    rgbaData[i * 4 + 0] = rgbData[i * 3 + 0];
    rgbaData[i * 4 + 1] = rgbData[i * 3 + 1];
    rgbaData[i * 4 + 2] = rgbData[i * 3 + 2];
    rgbaData[i * 4 + 3] = 1.0;
  }

  return rgbaData;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace tiff_utils {

////////////////////////////////////////////////////////////////////////////////////////////////////

uint32_t getNumLayers(std::string const& path) {
  auto* data = TIFFOpen(path.c_str(), "r");

  if (!data) {
    std::cerr << "Failed to open TIFF file'" << path << "' " << std::endl;
    return 0;
  }

  uint32_t numLayers = 0;

  do {
    numLayers++;
  } while (TIFFReadDirectory(data));

  TIFFClose(data);

  return numLayers;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

RGBATexture read2DTexture(std::string const& path, uint32_t layer) {
  RGBATexture texture;

  auto* data = TIFFOpen(path.c_str(), "r");

  if (!data) {
    std::cerr << "Failed to open TIFF file '" << path << "'" << std::endl;
    return texture;
  }

  TIFFGetField(data, TIFFTAG_IMAGELENGTH, &texture.height);
  TIFFGetField(data, TIFFTAG_IMAGEWIDTH, &texture.width);

  std::vector<float> rgbData(texture.width * texture.height * 3);

  TIFFSetDirectory(data, layer);
  for (unsigned y = 0; y < texture.height; y++) {
    TIFFReadScanline(data, &rgbData[texture.width * 3 * y], y);
  }

  TIFFClose(data);

  texture.data = addAlphaChannel(rgbData);

  return texture;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace tiff_utils
