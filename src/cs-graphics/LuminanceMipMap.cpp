////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "LuminanceMipMap.hpp"

#include "../cs-utils/FrameStats.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

namespace cs::graphics {

////////////////////////////////////////////////////////////////////////////////////////////////////

static const char* sComputeAverage = R"(
  #define LOCAL_SIZE 1024
  layout (local_size_x = LOCAL_SIZE, local_size_y = 1) in;

  #if NUM_MULTISAMPLES > 0
    layout (rgba32f, binding = 0) readonly uniform image2DMS uInHDRBuffer;
  #else
    layout (rgba32f, binding = 0) readonly uniform image2D uInHDRBuffer;
  #endif

  layout (binding = 1, rg32f) uniform image1D uOutLuminance;

  shared vec2 sData[LOCAL_SIZE];

  vec2 sampleHDRBuffer(ivec2 pos) {
    #if NUM_MULTISAMPLES > 0
      vec3 color = vec3(0.0);
      for (int i = 0; i < NUM_MULTISAMPLES; ++i) {
        color += imageLoad(uInHDRBuffer, ivec2(pos), i).rgb;
      }
      color /= NUM_MULTISAMPLES;
    #else
      vec3 color = imageLoad(uInHDRBuffer, ivec2(pos)).rgb;
    #endif
    float val = max(max(color.r, color.g), color.b);
    return vec2(val, val);
  }

  void main() {
    uint tid = gl_LocalInvocationID.x;
    uint gid = gl_GlobalInvocationID.x;
    ivec2 bufferSize = imageSize(uInHDRBuffer);

    uint i = gl_WorkGroupID.x * gl_WorkGroupSize.x * 2 + tid;

    if (i + gl_WorkGroupSize.x >= bufferSize.x * bufferSize.y) {
      return;
    }

    ivec2 leftIndex = ivec2(i / bufferSize.x, i % bufferSize.x);
    vec2 left = sampleHDRBuffer(leftIndex);

    ivec2 rightIndex = ivec2((i + gl_WorkGroupSize.x) / bufferSize.x, (i + gl_WorkGroupSize.x) % bufferSize.x);
    vec2 right = sampleHDRBuffer(rightIndex);

    sData[tid] = vec2(
        left.x + right.x,
        max(left.y, right.y)
    );

    memoryBarrierShared();
    barrier();

    for (uint s = gl_WorkGroupSize.x / 2; s > 32; s >>= 1) {
      if (tid < s) {
        vec2 left = sData[tid];
        vec2 right = sData[tid + s];
        sData[tid] = vec2(
            left.x + right.x,
            max(left.y, right.y)
        );
      }

      memoryBarrierShared();
      barrier();
    }

    if (tid < 32) {
      vec2 left = sData[tid];
      vec2 right = sData[tid + 32];
      left = vec2(
          left.x + right.x,
          max(left.y, right.y)
      );

      right = sData[tid + 16];
      left = vec2(
          left.x + right.x,
          max(left.y, right.y)
      );

      right = sData[tid + 8];
      left = vec2(
          left.x + right.x,
          max(left.y, right.y)
      );

      right = sData[tid + 4];
      left = vec2(
          left.x + right.x,
          max(left.y, right.y)
      );

      right = sData[tid + 2];
      left = vec2(
          left.x + right.x,
          max(left.y, right.y)
      );

      right = sData[tid + 1];
      sData[tid] = vec2(
          left.x + right.x,
          max(left.y, right.y)
      );
    }

    if (tid == 0) {
      imageStore(uOutLuminance, int(gl_WorkGroupID.x), vec4(sData[0].x, sData[0].y, 0.0, 0.0));
    }
  }
)";

////////////////////////////////////////////////////////////////////////////////////////////////////

constexpr int32_t BLOCK_SIZE = 1024;

LuminanceMipMap::LuminanceMipMap(uint32_t hdrBufferSamples, int hdrBufferWidth, int hdrBufferHeight)
    : mHDRBufferSamples(hdrBufferSamples)
    , mHDRBufferWidth(hdrBufferWidth)
    , mHDRBufferHeight(hdrBufferHeight) {

  mWorkGroups = static_cast<int>(std::ceil((mHDRBufferWidth * mHDRBufferHeight) / (2 * BLOCK_SIZE)));

  mLuminanceBuffer = std::make_unique<VistaTexture>(GL_TEXTURE_1D);
  mLuminanceBuffer->Bind();
  glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
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
  source += sComputeAverage;
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
    throw std::runtime_error(std::string("ERROR: Failed to compile shader\n") + log);
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

    throw std::runtime_error(std::string("ERROR: Failed to link compute shader\n") + log);
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

    for (size_t i = 0; i < mWorkGroups; ++i) {
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
