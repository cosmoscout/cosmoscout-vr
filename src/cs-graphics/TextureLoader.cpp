////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

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

    uint32 width{};
    uint32 height{};
    TIFFGetField(data, TIFFTAG_IMAGELENGTH, &height);
    TIFFGetField(data, TIFFTAG_IMAGEWIDTH, &width);

    uint16 bpp{};
    TIFFGetField(data, TIFFTAG_BITSPERSAMPLE, &bpp);

    int16 channels{};
    TIFFGetField(data, TIFFTAG_SAMPLESPERPIXEL, &channels);

    if (bpp != 8) {
      logger().error(
          "Failed to load '{}' with libtiff: Only 8 bit per sample are supported right now!",
          sFileName);
      return nullptr;
    }

    std::vector<char> pixels(width * height * channels);

    for (unsigned y = 0; y < height; y++) {
      TIFFReadScanline(data, &pixels[width * channels * y], y);
    }

    GLenum ePixelFormat = GL_RGBA;

    if (channels == 1) {
      ePixelFormat = GL_RED;
    } else if (channels == 2) {
      ePixelFormat = GL_RG;
    } else if (channels == 3) {
      ePixelFormat = GL_RGB;
    }

    result->UploadTexture(width, height, pixels.data(), true, ePixelFormat);

    TIFFClose(data);
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
