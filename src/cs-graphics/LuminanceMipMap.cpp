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
  #extension GL_KHR_shader_subgroup_basic : enable
  #extension GL_KHR_shader_subgroup_arithmetic : enable

  #define LOCAL_SIZE 1024
  layout (local_size_x = LOCAL_SIZE) in;

  #if NUM_MULTISAMPLES > 0
    layout (rgba32f, binding = 0) readonly uniform image2DMS uInHDRBuffer;
  #else
    layout (rgba32f, binding = 0) readonly uniform image2D uInHDRBuffer;
  #endif

  layout (rg32f, binding = 1) uniform image1D uOutLuminance;

  // Shared array for this work group. Contains total luminance in the x-component and max luminance
  // in the y-component of each element.
  shared vec2 sData[LOCAL_SIZE];

  // Returns the luminance for the pixel in a 2D vector.
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

  // Do a parallel reduction of the HDR buffer for the total and maximum luminance.
  void main() {
    uint tid = gl_LocalInvocationID.x;
    uint gid = gl_GlobalInvocationID.x;
    ivec2 bufferSize = imageSize(uInHDRBuffer);
    int maxSize = bufferSize.x * bufferSize.y;

    // 1. Step
    // We have half as many threads, as pixels on screen.
    // Each thread grabs two values from the HDR buffer.

    uint i = gl_WorkGroupID.x * gl_WorkGroupSize.x * 2 + tid;
    vec2 left = i < maxSize ? sampleHDRBuffer(ivec2(i % bufferSize.x, i / bufferSize.x)) : vec2(0);

    uint j = i + gl_WorkGroupSize.x;
    vec2 right = j < maxSize ? sampleHDRBuffer(ivec2(j % bufferSize.x, j / bufferSize.x)) : vec2(0);

    // The two values are being combined and written to this threads shared memory address.
    sData[tid] = vec2(
        left.x + right.x,
        max(left.y, right.y)
    );

    // Wait for all threads in the work group to finish.
    memoryBarrierShared();
    barrier();

    #ifdef GL_KHR_shader_subgroup_basic
      const uint subGroupSize = gl_SubgroupSize;
    #else
      // Default warp size for NVIDIA. AMD might have 32 or 64.
      const uint subGroupSize = 32;
    #endif

    // 2. Step
    // Do the actual parallel reduction.
    // Each thread combines its own value with a value of 2 * its current position.
    // Each loop the amount of working threads are halfed.
    // We stop, when only one warp is left.
    for (uint s = gl_WorkGroupSize.x / 2; s > subGroupSize; s >>= 1) {
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

    #if defined(GL_KHR_shader_subgroup_arithmetic) && defined(GL_KHR_shader_subgroup_basic)
      // We make use of special warp arithmetic to reduce the last warp.
      if (tid < subGroupSize) {
        vec2 value = sData[tid];
        float sum = subgroupAdd(value.x);
        float max = subgroupMax(value.y);
        if (subgroupElect()) {
            sData[tid] = vec2(sum, max);
        }
      }
    #else
      // Unroll the last warp for maximum performance gains.
      if (tid < 32) {
        vec2 left = sData[tid];
        vec2 right = sData[tid + 32];
        sData[tid] = vec2(
            left.x + right.x,
            max(left.y, right.y)
        );

        memoryBarrierShared();
        barrier();

        left = sData[tid];
        right = sData[tid + 16];
        sData[tid] = vec2(
            left.x + right.x,
            max(left.y, right.y)
        );

        memoryBarrierShared();
        barrier();

        left = sData[tid];
        right = sData[tid + 8];
        sData[tid] = vec2(
            left.x + right.x,
            max(left.y, right.y)
        );

        memoryBarrierShared();
        barrier();

        left = sData[tid];
        right = sData[tid + 4];
        sData[tid] = vec2(
            left.x + right.x,
            max(left.y, right.y)
        );

        memoryBarrierShared();
        barrier();

        left = sData[tid];
        right = sData[tid + 2];
        sData[tid] = vec2(
            left.x + right.x,
            max(left.y, right.y)
        );

        memoryBarrierShared();
        barrier();

        left = sData[tid];
        right = sData[tid + 1];
        sData[tid] = vec2(
            left.x + right.x,
            max(left.y, right.y)
        );
      }
    #endif

    // The first thread in each work group writes the final value to the output.
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
