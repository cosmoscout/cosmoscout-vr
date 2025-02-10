////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "TextureLoader.hpp"

#include "logger.hpp"

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include <stb_image.h>
#include <stb_image_resize.h>
#include <stb_image_write.h>
#undef STB_IMAGE_IMPLEMENTATION
#undef STB_IMAGE_WRITE_IMPLEMENTATION
#undef STB_IMAGE_RESIZE_IMPLEMENTATION

#include <VistaOGLExt/VistaOGLUtils.h>
#include <iostream>
#include <tiffio.h>
#include <vector>

namespace cs::graphics {

////////////////////////////////////////////////////////////////////////////////////////////////////

std::unique_ptr<VistaTexture> TextureLoader::loadFromFile(std::string const& sFileName) {

  std::string suffix = sFileName.substr(sFileName.rfind('.'));

  if (suffix == ".tga") {
    // load with vista
    logger().debug("Loading Texture '{}' with Vista.", sFileName);
    return std::unique_ptr<VistaTexture>(VistaOGLUtils::LoadTextureFromTga(sFileName));
  }

  std::unique_ptr<VistaTexture> result = std::make_unique<VistaTexture>(GL_TEXTURE_2D);

  if (suffix == ".tiff" || suffix == ".tif") {
    // load with tifflib
    logger().debug("Loading Texture '{}' with libtiff.", sFileName);

    auto* data = TIFFOpen(sFileName.c_str(), "r");
    if (!data) {
      logger().error("Failed to load '{}' with libtiff!", sFileName);
      return nullptr;
    }

    uint32_t width{};
    uint32_t height{};
    TIFFGetField(data, TIFFTAG_IMAGELENGTH, &height);
    TIFFGetField(data, TIFFTAG_IMAGEWIDTH, &width);

    uint16_t bpp{};
    TIFFGetField(data, TIFFTAG_BITSPERSAMPLE, &bpp);

    int16_t channels{};
    TIFFGetField(data, TIFFTAG_SAMPLESPERPIXEL, &channels);

    GLenum ePixelFormat = GL_RGBA;

    if (channels == 1) {
      ePixelFormat = GL_RED;
    } else if (channels == 2) {
      ePixelFormat = GL_RG;
    } else if (channels == 3) {
      ePixelFormat = GL_RGB;
    }

    if (bpp != 8 && bpp != 32) {
      logger().error(
          "Failed to load '{}' with libtiff: Only 8 or 32 bit per sample are supported right now!",
          sFileName);
      return nullptr;
    }

    if (bpp == 32) {

      std::vector<float> pixels(width * height * channels);

      for (unsigned y = 0; y < height; y++) {
        TIFFReadScanline(data, &pixels[width * channels * y], y);
      }

      result->Bind();
      glTexImage2D(
          GL_TEXTURE_2D, 0, GL_RGB32F, width, height, 0, ePixelFormat, GL_FLOAT, pixels.data());
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);

    } else {

      std::vector<char> pixels(width * height * channels);

      for (unsigned y = 0; y < height; y++) {
        TIFFReadScanline(data, &pixels[width * channels * y], y);
      }

      result->UploadTexture(width, height, pixels.data(), true, ePixelFormat);
    }

    TIFFClose(data);

  } else if (suffix == ".hdr") {

    // load with stb image
    logger().debug("Loading HDR Texture '{}' with stbi.", sFileName);

    int width{};
    int height{};
    int bpp{};
    int channels = 4;

    float* pixels = stbi_loadf(sFileName.c_str(), &width, &height, &bpp, channels);

    if (!pixels) {
      logger().error("Failed to load '{}' with stbi!", sFileName);
      return nullptr;
    }

    result->Bind();
    gluBuild2DMipmaps(GL_TEXTURE_2D, GL_RGBA32F, width, height, GL_RGBA, GL_FLOAT, pixels);

    stbi_image_free(pixels);

  } else {
    // load with stb image
    logger().debug("Loading Texture '{}' with stbi.", sFileName);

    int width{};
    int height{};
    int bpp{};
    int channels = 4;

    unsigned char* pixels = stbi_load(sFileName.c_str(), &width, &height, &bpp, channels);

    if (!pixels) {
      logger().error("Failed to load '{}' with stbi!", sFileName);
      return nullptr;
    }

    result->UploadTexture(width, height, pixels);

    stbi_image_free(pixels);
  }

  return result;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cs::graphics
