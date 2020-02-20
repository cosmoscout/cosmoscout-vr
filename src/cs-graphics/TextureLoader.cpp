////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "TextureLoader.hpp"

#include <VistaOGLExt/VistaOGLUtils.h>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image.h>
#include <stb_image_write.h>

#include <iostream>
#include <spdlog/spdlog.h>
#include <tiffio.h>
#include <vector>

namespace cs::graphics {

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace {

////////////////////////////////////////////////////////////////////////////////////////////////////

// constexpr GLenum stbi_component_to_format(int component)
inline GLenum stbi_component_to_format(int component) {
  switch (component) {
  case STBI_grey:
    return GL_RED;
  case STBI_grey_alpha:
    return GL_RG;
  case STBI_rgb:
    return GL_RGB;
  case STBI_rgb_alpha:
    return GL_RGBA;
  default:
    return GL_RGB;
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// constexpr GLint stbi_component_to_internal_format(int component)
inline GLint stbi_component_to_internal_format(int component) {
  switch (component) {
  case STBI_grey:
    return GL_RED;
  case STBI_grey_alpha:
    return GL_RG;
  case STBI_rgb:
    return GL_RGB;
  case STBI_rgb_alpha:
    return GL_RGBA;
  default:
    return GL_RGB;
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace

////////////////////////////////////////////////////////////////////////////////////////////////////

std::shared_ptr<VistaTexture> TextureLoader::loadFromFile(std::string const& sFileName) {

  std::string suffix = sFileName.substr(sFileName.rfind('.'));

  if (suffix == ".tga") {
    // load with vista
    spdlog::debug("Loading Texture '{}' with Vista.", sFileName);
    return std::shared_ptr<VistaTexture>(VistaOGLUtils::LoadTextureFromTga(sFileName));
  }

  std::shared_ptr<VistaTexture> result(new VistaTexture(GL_TEXTURE_2D));

  if (suffix == ".tiff" || suffix == ".tif") {
    // load with tifflib
    spdlog::debug("Loading Texture '{}' with libtiff.", sFileName);

    auto data = TIFFOpen(sFileName.c_str(), "r");
    if (!data) {
      spdlog::error("Failed to load '{}' with libtiff!", sFileName);
      return nullptr;
    }

    uint32 width, height;
    TIFFGetField(data, TIFFTAG_IMAGELENGTH, &height);
    TIFFGetField(data, TIFFTAG_IMAGEWIDTH, &width);

    uint16 bpp;
    TIFFGetField(data, TIFFTAG_BITSPERSAMPLE, &bpp);

    int16 channels;
    TIFFGetField(data, TIFFTAG_SAMPLESPERPIXEL, &channels);

    if (bpp != 8) {
      spdlog::error(
          "Failed to load '{}' with libtiff: Only 8 bit per sample are supported right now!",
          sFileName);
      return nullptr;
    }

    std::vector<char> pixels(width * height * channels);

    for (unsigned y = 0; y < height; y++) {
      TIFFReadScanline(data, &pixels[width * channels * y], y);
    }

    GLenum ePixelFormat = GL_RGBA;

    if (channels == 1)
      ePixelFormat = GL_RED;
    else if (channels == 2)
      ePixelFormat = GL_RG;
    else if (channels == 3)
      ePixelFormat = GL_RGB;

    result->UploadTexture(width, height, pixels.data(), true, ePixelFormat);

    TIFFClose(data);
  } else {
    // load with stb image
    spdlog::debug("Loading Texture '{}' with stbi.", sFileName);

    int width, height, bpp;
    int channels = 4;

    unsigned char* pixels = stbi_load(sFileName.c_str(), &width, &height, &bpp, channels);

    if (!pixels) {
      spdlog::error("Failed to load '{}' with stbi!", sFileName);
      return nullptr;
    }

    result->UploadTexture(width, height, pixels);

    stbi_image_free(pixels);
  }

  return result;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cs::graphics
