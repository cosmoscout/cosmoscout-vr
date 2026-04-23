////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "utils.hpp"

#include "logger.hpp"

#include <tiffio.h>

namespace csp::atmospheres::utils {

////////////////////////////////////////////////////////////////////////////////////////////////////

std::tuple<GLuint, glm::ivec2> read2DTexture(std::string const& path) {
  auto* data = TIFFOpen(path.c_str(), "r");

  if (!data) {
    logger().error("Failed to open TIFF file '{}'", path);
    return {0u, {0, 0}};
  }

  uint32_t width{};
  uint32_t height{};

  TIFFGetField(data, TIFFTAG_IMAGELENGTH, &height);
  TIFFGetField(data, TIFFTAG_IMAGEWIDTH, &width);

  std::vector<float> pixels(width * height * 3);

  for (unsigned y = 0; y < height; y++) {
    TIFFReadScanline(data, &pixels[width * 3 * y], y);
  }

  TIFFClose(data);

  GLuint texture;
  glGenTextures(1, &texture);
  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, texture);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, width, height, 0, GL_RGB, GL_FLOAT, pixels.data());

  return {texture, {width, height}};
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::tuple<GLuint, glm::ivec3> read3DTexture(std::string const& path) {
  auto* data = TIFFOpen(path.c_str(), "r");

  if (!data) {
    logger().error("Failed to open TIFF file '{}'", path);
    return {0u, {0, 0, 0}};
  }

  uint32_t width{};
  uint32_t height{};
  uint32_t depth{};

  TIFFGetField(data, TIFFTAG_IMAGELENGTH, &height);
  TIFFGetField(data, TIFFTAG_IMAGEWIDTH, &width);

  do {
    depth++;
  } while (TIFFReadDirectory(data));

  std::vector<float> pixels(width * height * depth * 3);

  for (unsigned z = 0; z < depth; z++) {
    TIFFSetDirectory(data, z);
    for (unsigned y = 0; y < height; y++) {
      TIFFReadScanline(data, &pixels[width * 3 * y + (3 * width * height * z)], y);
    }
  }

  TIFFClose(data);

  GLuint texture;
  glGenTextures(1, &texture);
  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_3D, texture);
  glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
  glTexImage3D(
      GL_TEXTURE_3D, 0, GL_RGB32F, width, height, depth, 0, GL_RGB, GL_FLOAT, pixels.data());

  return {texture, {width, height, depth}};
}

std::vector<float> utils::readTexture(std::string const& path, int *width, int *height, int *channels) {
  float *data = stbi_loadf(path.c_str(), width, height, channels, 0);
  int size = *width * *height * *channels;

  vstr::debug() << "Loading '" << path << "' with width = " << *width << ", height = " << *height
    << ", channels = " << *channels << "." << std::endl;

  if (!data) {
    vstr::err() << "Failed to load texture at '" << path << "'." << std::endl;
  }

  std::vector<float> dataVec(data, data + size);
  stbi_image_free(data);

  return dataVec;
}

void storeShaderInfoLog(std::string shaderName, GLuint shaderId) {
  GLint iCompileSuccess = 0;
  glGetShaderiv(shaderId, GL_COMPILE_STATUS, &iCompileSuccess);

  if (iCompileSuccess != GL_TRUE) {
    vstr::errp() << "Failed to compile '" << shaderName << "'." << std::endl;
  }

  int iLogSize;
  glGetShaderiv(shaderId, GL_INFO_LOG_LENGTH, &iLogSize);

  if (iLogSize > 0) {
    vstr::debug() << "Writing shader info log to shader-log.txt..." << std::endl;

    char* sLog = new char[iLogSize];
    glGetShaderInfoLog(shaderId, iLogSize, NULL, sLog);
    
    std::ofstream logFile;
    logFile.open ("shader-log.txt", std::ios::in | std::ios::trunc);
    if (logFile.is_open()) {
      logFile << "Failed to compile shader program: " << std::endl;
      logFile << shaderName << " (ID = " << shaderId << ")" << std::endl;
      logFile << sLog << std::endl << std::endl;
      logFile.close();
    } else {
      vstr::errp() << "Failed to write shader info log." << std::endl;
    }

    delete[] sLog;
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::atmospheres::utils
