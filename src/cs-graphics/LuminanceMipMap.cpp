////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "LuminanceMipMap.hpp"

#include "../cs-utils/FrameStats.hpp"

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

namespace cs::graphics {

////////////////////////////////////////////////////////////////////////////////////////////////////

LuminanceMipMap::LuminanceMipMap(uint32_t hdrBufferSamples, int hdrBufferWidth, int hdrBufferHeight)
    : mHDRBufferSamples(hdrBufferSamples)
    , mHDRBufferWidth(hdrBufferWidth)
    , mHDRBufferHeight(hdrBufferHeight) {
  GLint       subgroupSize = 32;
  std::string shaderSource = "../share/resources/shaders/computeLuminanceClassic.comp";
  if (glewIsSupported("GL_KHR_shader_subgroup")) {
    glGetIntegerv(GL_SUBGROUP_SIZE_KHR, &subgroupSize);
    shaderSource = "../share/resources/shaders/computeLuminanceFast.comp";
  }
  GLint workgroupSize = subgroupSize * subgroupSize;

  mWorkGroups =
      static_cast<int>(std::ceil((mHDRBufferWidth * mHDRBufferHeight) / (2.0 * workgroupSize)));

  mLuminanceBuffer = std::make_unique<VistaTexture>(GL_TEXTURE_1D);
  mLuminanceBuffer->Bind();
  glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);

  glTexStorage1D(GL_TEXTURE_1D, 1, GL_RG32F, mWorkGroups);

  // Create pixel buffer object for luminance read-back.
  glGenBuffers(1, &mPBO);
  glBindBuffer(GL_PIXEL_PACK_BUFFER, mPBO);
  glBufferStorage(GL_PIXEL_PACK_BUFFER, mWorkGroups * sizeof(float) * 2, nullptr, GL_MAP_READ_BIT);
  glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);

  // Create the compute shader.
  auto        shader = glCreateShader(GL_COMPUTE_SHADER);
  std::string source = "#version 430\n";
  source += "#define NUM_MULTISAMPLES " + std::to_string(mHDRBufferSamples) + "\n";
  source += "#define WORKGROUP_SIZE " + std::to_string(workgroupSize) + "\n";
  source += "#define SUBGROUP_SIZE " + std::to_string(subgroupSize) + "\n";
  source += cs::utils::filesystem::loadToString(shaderSource);
  const char* pSource = source.c_str();
  glShaderSource(shader, 1, &pSource, nullptr);
  glCompileShader(shader);

  int rvalue = 0;
  glGetShaderiv(shader, GL_COMPILE_STATUS, &rvalue);
  if (rvalue != GL_TRUE) {
    auto log_length = 0;
    glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &log_length);
    std::vector<char> v(log_length);
    glGetShaderInfoLog(shader, log_length, nullptr, v.data());
    std::string log(begin(v), end(v));
    glDeleteShader(shader);
    std::string error = std::string("Failed to compile shader\n") + log;
    logger().critical(error);
    throw std::runtime_error("ERROR: " + error);
  }

  mComputeProgram = glCreateProgram();
  glAttachShader(mComputeProgram, shader);
  glLinkProgram(mComputeProgram);
  glDeleteShader(shader);

  glGetProgramiv(mComputeProgram, GL_LINK_STATUS, &rvalue);
  if (rvalue != GL_TRUE) {
    auto log_length = 0;
    glGetProgramiv(mComputeProgram, GL_INFO_LOG_LENGTH, &log_length);
    std::vector<char> v(log_length);
    glGetProgramInfoLog(mComputeProgram, log_length, nullptr, v.data());
    std::string log(begin(v), end(v));
    std::string error = std::string("Failed to link compute shader\n") + log;
    logger().critical(error);
    throw std::runtime_error("ERROR: " + error);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

LuminanceMipMap::~LuminanceMipMap() {
  glDeleteBuffers(1, &mPBO);
  glDeleteProgram(mComputeProgram);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void LuminanceMipMap::update(VistaTexture* hdrBufferComposite) {
  utils::FrameStats::ScopedTimer timer("Compute Scene Luminance");

  // Read the luminance values from the last frame. ------------------------------------------------
  glBindBuffer(GL_PIXEL_PACK_BUFFER, mPBO);

  // Map the pixel buffer object and read the two values.
  if (mDataAvailable) {
    glm::vec2* data = static_cast<glm::vec2*>(glMapBuffer(GL_PIXEL_PACK_BUFFER, GL_READ_ONLY));

    mLastTotalLuminance   = 0;
    mLastMaximumLuminance = 0;

    for (int i = 0; i < mWorkGroups; ++i) {
      glm::vec2 value = data[i];
      mLastTotalLuminance += std::isnan(value.x) ? 0.F : value.x;
      mLastMaximumLuminance = std::max(mLastMaximumLuminance, std::isnan(value.y) ? 0.F : value.y);
    }

    glUnmapBuffer(GL_PIXEL_PACK_BUFFER);
  }

  // Update current luminance mipmap. --------------------------------------------------------------

  glUseProgram(mComputeProgram);
  glBindImageTexture(0, hdrBufferComposite->GetId(), 0, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA32F);
  glBindImageTexture(1, mLuminanceBuffer->GetId(), 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RG32F);

  // Make sure writing has finished.
  glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

  glDispatchCompute(mWorkGroups, 1, 1);

  glBindImageTexture(0, 0, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RG32F);
  glMemoryBarrier(GL_PIXEL_BUFFER_BARRIER_BIT);

  mLuminanceBuffer->Bind();
  glGetTexImage(GL_TEXTURE_1D, 0, GL_RG, GL_FLOAT, nullptr);
  mLuminanceBuffer->Unbind();

  glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);

  mDataAvailable = true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool LuminanceMipMap::getIsDataAvailable() const {
  return mDataAvailable;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

float LuminanceMipMap::getLastTotalLuminance() const {
  return mLastTotalLuminance;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

float LuminanceMipMap::getLastMaximumLuminance() const {
  return mLastMaximumLuminance;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cs::graphics
