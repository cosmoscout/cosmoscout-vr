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

void write2D(std::string const& path, float* texture, int width, int height, int components) {
  auto* tiff = TIFFOpen(path.c_str(), "w");
  TIFFSetField(tiff, TIFFTAG_IMAGEWIDTH, width);
  TIFFSetField(tiff, TIFFTAG_IMAGELENGTH, height);
  TIFFSetField(tiff, TIFFTAG_SAMPLESPERPIXEL, components);
  TIFFSetField(tiff, TIFFTAG_BITSPERSAMPLE, 32);
  TIFFSetField(tiff, TIFFTAG_SAMPLEFORMAT, SAMPLEFORMAT_IEEEFP);
  TIFFSetField(tiff, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_RGB);
  TIFFSetField(tiff, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
  TIFFSetField(tiff, TIFFTAG_ROWSPERSTRIP, 1);
  for (int y = 0; y < height; ++y) {
    TIFFWriteScanline(tiff, texture + y * width * components, height - y - 1);
  }
  TIFFClose(tiff);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void write3D(
    std::string const& path, float* texture, int width, int height, int depth, int components) {
  auto* tiff = TIFFOpen(path.c_str(), "w");

  for (int z = 0; z < depth; ++z) {
    TIFFSetField(tiff, TIFFTAG_PAGENUMBER, z, z);
    TIFFSetField(tiff, TIFFTAG_SUBFILETYPE, FILETYPE_PAGE);
    TIFFSetField(tiff, TIFFTAG_IMAGEWIDTH, width);
    TIFFSetField(tiff, TIFFTAG_IMAGELENGTH, height);
    TIFFSetField(tiff, TIFFTAG_SAMPLESPERPIXEL, components);
    TIFFSetField(tiff, TIFFTAG_BITSPERSAMPLE, 32);
    TIFFSetField(tiff, TIFFTAG_SAMPLEFORMAT, SAMPLEFORMAT_IEEEFP);
    TIFFSetField(tiff, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_RGB);
    TIFFSetField(tiff, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
    TIFFSetField(tiff, TIFFTAG_ROWSPERSTRIP, 1);
    for (int y = 0; y < height; ++y) {
      TIFFWriteScanline(
          tiff, texture + z * width * height * components + y * width * components, height - y - 1);
    }
    TIFFWriteDirectory(tiff);
  }
  TIFFClose(tiff);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace tiff_utils
