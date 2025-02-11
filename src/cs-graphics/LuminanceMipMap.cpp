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

  #if NUM_MULTISAMPLES > 0
    layout (rgba32f, binding = 0) readonly uniform image2DMS uInHDRBuffer;
  #else
    layout (rgba32f, binding = 0) readonly uniform image2D uInHDRBuffer;
  #endif

  layout (rg32f, binding = 1) uniform image1D uOutLuminance;

  // Returns the luminance for the pixel.
  float sampleHDRBuffer(ivec2 pos) {
    #if NUM_MULTISAMPLES > 0
      vec3 color = vec3(0.0);
      for (int i = 0; i < NUM_MULTISAMPLES; ++i) {
        color += imageLoad(uInHDRBuffer, pos, i).rgb;
      }
      color /= NUM_MULTISAMPLES;
    #else
      vec3 color = imageLoad(uInHDRBuffer, pos).rgb;
    #endif
    return max(max(color.r, color.g), color.b);
  }

  ivec2 indexToPos(uint index, int width) {
    return ivec2(index % width, index / width);
  }

  // We have support for subgroups and can do a lot of optimizations!
  #if defined(GL_KHR_shader_subgroup_arithmetic) && defined(GL_KHR_shader_subgroup_basic)
    // The workgroup size will be subgroupSize squared.
    layout (local_size_x = WORKGROUP_SIZE) in;

    // Each subgroup gets a space in shared memory.
    shared float sSum[SUBGROUP_SIZE];
    shared float sMax[SUBGROUP_SIZE];

    // This shader makes great use of subgroup optimization. We will have subgroupSize squared
    // threads. This allows us to calculate the sum and max in just three steps:
    // 1. Step: Each thread grabs two values from the HDR buffer
    // 2. Step: Each subgroup reduces their values and write their single result into shared memory.
    // 3. Step: The first subgroup fetches the values from shared memory and reduces them to a
    //          single value. This value will be written to the output buffer.
    void main() {
      ivec2 bufferSize = imageSize(uInHDRBuffer);
      int   maxSize    = bufferSize.x * bufferSize.y;

      // 1. Step
      // Each thread grabs two values from the HDR buffer. We need to be careful with the indexing
      // here, so that neighboring threads access neighboring indices in both calls.
      uint  i    = gl_WorkGroupID.x * gl_WorkGroupSize.x * 2 + gl_LocalInvocationID.x;
      float left = i < maxSize ? sampleHDRBuffer(indexToPos(i, bufferSize.x)) : 0;

      uint  j     = i + gl_WorkGroupSize.x;
      float right = j < maxSize ? sampleHDRBuffer(indexToPos(j, bufferSize.x)) : 0;

      // 2. Step
      // All threads in a subgroup will sum their values up and calculate their max.
      float initialSum = subgroupAdd(left + right);
      float initialMax = subgroupMax(max(left, right));

      // The subgroup max and sum are being combined and written to this subgroups shared memory
      // address.
      if (subgroupElect()) {
        sSum[gl_SubgroupID] = initialSum;
        sMax[gl_SubgroupID] = initialMax;
      }

      // Wait for all threads in the work group to finish.
      memoryBarrierShared();
      barrier();

      // 3. Step
      // The lowest indexed subgroup grab the remaining values from shared memory and reduce them.
      // The result is written to the output buffer at the location of this work groups id.
      if (gl_SubgroupID == 0) {
        float sum = subgroupAdd(sSum[gl_SubgroupInvocationID]);
        float max = subgroupMax(sMax[gl_SubgroupInvocationID]);
        if (subgroupElect()) {
            imageStore(uOutLuminance, int(gl_WorkGroupID.x), vec4(sum, max, 0.0, 0.0));
        }
      }
    }

  #else // We don't have support for subgroups and do a standard parallel reduction!
    layout (local_size_x = 1024) in;

    shared float sSum[1024];
    shared float sMax[1024];

    // This shader does a standard parallel reduction based on a CUDA webinar:
    // https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
    void main() {
      uint  tid        = gl_LocalInvocationID.x;
      ivec2 bufferSize = imageSize(uInHDRBuffer);
      int   maxSize    = bufferSize.x * bufferSize.y;

      // 1. Step
      // We have half as many threads, as pixels on screen.
      // Each thread grabs two values from the HDR buffer.

      uint  i    = gl_WorkGroupID.x * gl_WorkGroupSize.x * 2 + tid;
      float left = i < maxSize ? sampleHDRBuffer(indexToPos(i, bufferSize.x)) : 0;

      uint  j     = i + gl_WorkGroupSize.x;
      float right = j < maxSize ? sampleHDRBuffer(indexToPos(j, bufferSize.x)) : 0;

      // The two values are being combined and written to this threads shared memory address.
      sSum[tid] = left + right;
      sMax[tid] = max(left, right);

      // Wait for all threads in the work group to finish.
      memoryBarrierShared();
      barrier();

      // 2. Step
      // Each thread combines its own value with a value of 2 times its current position.
      // We will halve the amount of threads each turn.
      // We repeat this step until one warp is left.
      // We could do this in a loop, but manual unrolling is faster (I profiled this!).
      if (tid < 256) {
        sSum[tid] += sSum[tid + 256];
        sMax[tid]  = max(sMax[tid], sMax[tid + 256]);
      }

      memoryBarrierShared();
      barrier();

      if (tid < 128) {
        sSum[tid] += sSum[tid + 128];
        sMax[tid]  = max(sMax[tid], sMax[tid + 128]);
      }

      memoryBarrierShared();
      barrier();

      if (tid < 64) {
        sSum[tid] += sSum[tid + 64];
        sMax[tid]  = max(sMax[tid], sMax[tid + 64]);
      }

      memoryBarrierShared();
      barrier();

      // 3. Step
      // We don't need to check the thread id for the last warp, since they are more efficient
      // doing the same work.
      if (tid < 32) {
        sSum[tid] += sSum[tid + 32];
        sMax[tid]  = max(sMax[tid], sMax[tid + 32]);
        memoryBarrierShared();
        barrier();

        sSum[tid] += sSum[tid + 16];
        sMax[tid]  = max(sMax[tid], sMax[tid + 16]);
        memoryBarrierShared();
        barrier();

        sSum[tid] += sSum[tid + 8];
        sMax[tid]  = max(sMax[tid], sMax[tid + 8]);
        memoryBarrierShared();
        barrier();

        sSum[tid] += sSum[tid + 4];
        sMax[tid]  = max(sMax[tid], sMax[tid + 4]);
        memoryBarrierShared();
        barrier();

        sSum[tid] += sSum[tid + 2];
        sMax[tid]  = max(sMax[tid], sMax[tid + 2]);
        memoryBarrierShared();
        barrier();

        float sum = sSum[tid] + sSum[tid + 1];
        float max = max(sMax[tid], sMax[tid + 1]);

        // The first thread in each work group writes the final value to the output.
        if (tid == 0) {
          imageStore(uOutLuminance, int(gl_WorkGroupID.x), vec4(sum, max, 0.0, 0.0));
        }
      }
    }
  #endif
)";

////////////////////////////////////////////////////////////////////////////////////////////////////

LuminanceMipMap::LuminanceMipMap(uint32_t hdrBufferSamples, int hdrBufferWidth, int hdrBufferHeight)
    : mHDRBufferSamples(hdrBufferSamples)
    , mHDRBufferWidth(hdrBufferWidth)
    , mHDRBufferHeight(hdrBufferHeight) {

  GLint subgroupSize = 32;
  if (glewIsSupported("GL_KHR_shader_subgroup")) {
    glGetIntegerv(GL_SUBGROUP_SIZE_KHR, &subgroupSize);
  }
  GLint workgroupSize = subgroupSize * subgroupSize;

  mWorkGroups = static_cast<int>(std::ceil((mHDRBufferWidth * mHDRBufferHeight) / (2.0 * workgroupSize)));

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

    for (int i = 0; i < mWorkGroups; ++i) {
      glm::vec2 value = data[i];
      mLastTotalLuminance  += std::isnan(value.x) ? 0.F : value.x;
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
