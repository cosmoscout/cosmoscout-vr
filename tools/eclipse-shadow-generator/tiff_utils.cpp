////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "tiff_utils.hpp"

#include <iostream>
#include <tiffio.h>

namespace tiff_utils {

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

RGBATexture read2DTexture(std::string const& path) {
  RGBATexture texture;

  auto* data = TIFFOpen(path.c_str(), "r");

  if (!data) {
    std::cerr << "Failed to open TIFF file '" << path << "'" << std::endl;
    return texture;
  }

  TIFFGetField(data, TIFFTAG_IMAGELENGTH, &texture.height);
  TIFFGetField(data, TIFFTAG_IMAGEWIDTH, &texture.width);
  texture.depth = 1;

  std::vector<float> rgbData(texture.width * texture.height * 3);

  for (unsigned y = 0; y < texture.height; y++) {
    TIFFReadScanline(data, &rgbData[texture.width * 3 * y], y);
  }

  TIFFClose(data);

  texture.data = addAlphaChannel(rgbData);

  return texture;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

RGBATexture read3DTexture(std::string const& path) {
  RGBATexture texture;

  auto* data = TIFFOpen(path.c_str(), "r");

  if (!data) {
    std::cerr << "Failed to open TIFF file '" << path << "'" << std::endl;
    return texture;
  }

  TIFFGetField(data, TIFFTAG_IMAGELENGTH, &texture.height);
  TIFFGetField(data, TIFFTAG_IMAGEWIDTH, &texture.width);

  do {
    texture.depth++;
  } while (TIFFReadDirectory(data));

  std::vector<float> rgbData(texture.width * texture.height * texture.depth * 3);

  for (unsigned z = 0; z < texture.depth; z++) {
    TIFFSetDirectory(data, z);
    for (unsigned y = 0; y < texture.height; y++) {
      TIFFReadScanline(
          data, &rgbData[texture.width * 3 * y + (3 * texture.width * texture.height * z)], y);
    }
  }

  TIFFClose(data);

  texture.data = addAlphaChannel(rgbData);

  return texture;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace tiff_utils
