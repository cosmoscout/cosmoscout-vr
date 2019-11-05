////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "LuminanceMipMap.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

namespace cs::graphics {

////////////////////////////////////////////////////////////////////////////////////////////////////

const std::string LuminanceMipMap::sComputeAverage = R"(
  #version 430
  
  layout (local_size_x = 16, local_size_y = 16) in;

  layout (rg32f,   binding = 0) writeonly uniform image2D uOutLuminance;
  layout (rgba32f, binding = 1) readonly  uniform image2D uInHDRBuffer;
  layout (rg32f,   binding = 2) readonly  uniform image2D uInLuminance;

  uniform int uLevel;

  void sampleHDRBuffer(inout float oMaximumLuminance, inout float oTotalLuminance, ivec2 offset) {
    vec3 color = imageLoad(uInHDRBuffer, ivec2(gl_GlobalInvocationID.xy*2 + offset)).rgb;
    float val = max(max(color.r, color.g), color.b);
    oTotalLuminance += val;
    oMaximumLuminance = max(oMaximumLuminance, val);
  }

  void samplePyramid(inout float oMaximumLuminance, inout float oTotalLuminance, ivec2 offset) {
    vec2 totalMax = imageLoad(uInLuminance, ivec2(gl_GlobalInvocationID.xy*2 + offset)).rg;
    oTotalLuminance += totalMax.r;
    oMaximumLuminance = max(oMaximumLuminance, totalMax.g);
  }

  void main() {
    ivec2 storePos = ivec2(gl_GlobalInvocationID.xy);
    ivec2 size     = imageSize(uOutLuminance);

    if (storePos.x >= size.x || storePos.y >= size.y) {
      return;
    }

    float oTotalLuminance = 0;
    float oMaximumLuminance = 0;
    
    if (uLevel == 0) {
      sampleHDRBuffer(oMaximumLuminance, oTotalLuminance, ivec2(0, 0));
      sampleHDRBuffer(oMaximumLuminance, oTotalLuminance, ivec2(0, 1));
      sampleHDRBuffer(oMaximumLuminance, oTotalLuminance, ivec2(1, 0));
      sampleHDRBuffer(oMaximumLuminance, oTotalLuminance, ivec2(1, 1));
    } else {
      samplePyramid(oMaximumLuminance, oTotalLuminance, ivec2(0, 0));
      samplePyramid(oMaximumLuminance, oTotalLuminance, ivec2(0, 1));
      samplePyramid(oMaximumLuminance, oTotalLuminance, ivec2(1, 0));
      samplePyramid(oMaximumLuminance, oTotalLuminance, ivec2(1, 1));

      // handle cases close to right and top edge
      ivec2 maxCoords = imageSize(uInLuminance) - ivec2(1);
      if (gl_GlobalInvocationID.x*2 == maxCoords.x - 2) {
        samplePyramid(oMaximumLuminance, oTotalLuminance, ivec2(2,0));
        samplePyramid(oMaximumLuminance, oTotalLuminance, ivec2(2,1));
      }

      if (gl_GlobalInvocationID.y*2 == maxCoords.y - 2) {
        samplePyramid(oMaximumLuminance, oTotalLuminance, ivec2(0,2));
        samplePyramid(oMaximumLuminance, oTotalLuminance, ivec2(1,2));

        if (gl_GlobalInvocationID.x*2 == maxCoords.x - 2) {
          samplePyramid(oMaximumLuminance, oTotalLuminance, ivec2(2,2));
        }
      }
    }

    imageStore(uOutLuminance, storePos, vec4(oTotalLuminance, oMaximumLuminance, 0.0, 0.0));
  }
)";

////////////////////////////////////////////////////////////////////////////////////////////////////

LuminanceMipMap::LuminanceMipMap(int hdrBufferWidth, int hdrBufferHeight)
    : VistaTexture(GL_TEXTURE_2D)
    , mHDRBufferWidth(hdrBufferWidth)
    , mHDRBufferHeight(hdrBufferHeight) {
  // create luminance mipmap storage
  Bind();

  int iWidth  = mHDRBufferWidth / 2;
  int iHeight = mHDRBufferHeight / 2;

  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

  mMaxLevels = std::max(1.0, std::floor(std::log2(std::max(iWidth, iHeight))) + 1);

  glTexStorage2D(GL_TEXTURE_2D, mMaxLevels, GL_RG32F, iWidth, iHeight);

  // create pixel buffer object for luminance read-back
  glGenBuffers(1, &mPBO);
  glBindBuffer(GL_PIXEL_PACK_BUFFER, mPBO);
  glBufferStorage(GL_PIXEL_PACK_BUFFER, sizeof(float) * 2, nullptr, GL_MAP_READ_BIT);

  mDataAvailable = false;

  // create compute shader
  auto        shader = glCreateShader(GL_COMPUTE_SHADER);
  const char* c_str  = sComputeAverage.c_str();
  glShaderSource(shader, 1, &c_str, nullptr);
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

    std::cout << (std::string("ERROR: Failed to link compute shader\n") + log) << std::endl;
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

LuminanceMipMap::~LuminanceMipMap() {
  glDeleteBuffers(1, &mPBO);
  glDeleteProgram(mComputeProgram);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void LuminanceMipMap::update(ExposureMeteringMode meteringMode, VistaTexture* hdrBufferComposite) {
  int iWidth  = mHDRBufferWidth / 2;
  int iHeight = mHDRBufferHeight / 2;

  // read luminance from last frame -------------------------------------------------------------
  glBindBuffer(GL_PIXEL_PACK_BUFFER, mPBO);
  if (mDataAvailable) {
    float* data           = (float*)glMapBuffer(GL_PIXEL_PACK_BUFFER, GL_READ_ONLY);
    mLastTotalLuminance   = data[0];
    mLastMaximumLuminance = data[1];
    glUnmapBuffer(GL_PIXEL_PACK_BUFFER);

    if (std::isnan(mLastTotalLuminance)) {
      mLastTotalLuminance = 0.0;
    }

    if (std::isnan(mLastMaximumLuminance)) {
      mLastMaximumLuminance = 0.0;
    }
  }

  // update current luminance mipmap ------------------------------------------------------------

  glUseProgram(mComputeProgram);

  glBindImageTexture(1, hdrBufferComposite->GetId(), 0, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA32F);

  for (int i(0); i < mMaxLevels; ++i) {
    int width  = std::max(1.0, std::floor(static_cast<double>(iWidth) / std::pow(2, i)));
    int height = std::max(1.0, std::floor(static_cast<double>(iHeight) / std::pow(2, i)));

    glUniform1i(glGetUniformLocation(mComputeProgram, "uLevel"), i);
    glBindImageTexture(0, GetId(), i, GL_FALSE, 0, GL_WRITE_ONLY, GL_RG32F);

    if (i > 0) {
      glBindImageTexture(2, GetId(), i - 1, GL_FALSE, 0, GL_READ_ONLY, GL_RG32F);
    }

    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
    glDispatchCompute(std::ceil(1.0 * width / 16), std::ceil(1.0 * height / 16), 1);
  }

  glBindImageTexture(0, 0, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RG32F);
  glMemoryBarrier(GL_PIXEL_BUFFER_BARRIER_BIT);

  // copy to PBO ---------------------------------------------------------------------------------
  Bind();
  glGetTexImage(GL_TEXTURE_2D, mMaxLevels - 1, GL_RG, GL_FLOAT, 0);
  Unbind();
  glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);

  mLastMeteringMode = meteringMode;
  mDataAvailable    = true;
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
