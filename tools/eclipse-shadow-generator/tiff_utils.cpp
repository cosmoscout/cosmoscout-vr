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

RGBTexture read2DTexture(std::string const& path) {
  RGBTexture texture;

  auto* data = TIFFOpen(path.c_str(), "r");

  if (!data) {
    std::cerr << "Failed to open TIFF file '" << path << "'" << std::endl;
    return texture;
  }

  TIFFGetField(data, TIFFTAG_IMAGELENGTH, &texture.height);
  TIFFGetField(data, TIFFTAG_IMAGEWIDTH, &texture.width);
  texture.depth = 1;

  texture.data.resize(texture.width * texture.height * 3);

  for (unsigned y = 0; y < texture.height; y++) {
    TIFFReadScanline(data, &texture.data[texture.width * 3 * y], y);
  }

  TIFFClose(data);

  return texture;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

RGBTexture read3DTexture(std::string const& path) {
  RGBTexture texture;

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

  texture.data.resize(texture.width * texture.height * texture.depth * 3);

  for (unsigned z = 0; z < texture.depth; z++) {
    TIFFSetDirectory(data, z);
    for (unsigned y = 0; y < texture.height; y++) {
      TIFFReadScanline(
          data, &texture.data[texture.width * 3 * y + (3 * texture.width * texture.height * z)], y);
    }
  }

  TIFFClose(data);

  return texture;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace tiff_utils
