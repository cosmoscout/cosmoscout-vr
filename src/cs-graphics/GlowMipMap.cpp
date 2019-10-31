////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "GlowMipMap.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

namespace cs::graphics {

////////////////////////////////////////////////////////////////////////////////////////////////////

const std::string GlowMipMap::sGlowShader = R"(
  #version 430
  
  layout (local_size_x = 16, local_size_y = 16) in;

  layout (rgba32f, binding = 0) writeonly uniform image2D uOutColor;
  layout (rgba32f, binding = 1) readonly  uniform image2D uInColor;

  uniform int uPass;
  uniform float uThreshold;

  vec3 sampleHigherLevel(ivec2 offset) {
    vec3 col = imageLoad(uInColor, ivec2((gl_GlobalInvocationID.xy+offset)*2 + ivec2(0,0))).rgb * 0.25
             + imageLoad(uInColor, ivec2((gl_GlobalInvocationID.xy+offset)*2 + ivec2(1,0))).rgb * 0.25
             + imageLoad(uInColor, ivec2((gl_GlobalInvocationID.xy+offset)*2 + ivec2(0,1))).rgb * 0.25
             + imageLoad(uInColor, ivec2((gl_GlobalInvocationID.xy+offset)*2 + ivec2(1,1))).rgb * 0.25;
    return col;
  }

  vec3 sampleSameLevel(ivec2 offset) {
    return imageLoad(uInColor, ivec2(gl_GlobalInvocationID.xy+offset)).rgb;
  }

  void main() {
    ivec2 storePos = ivec2(gl_GlobalInvocationID.xy);
    ivec2 size     = imageSize(uOutColor);

    if (storePos.x >= size.x || storePos.y >= size.y) {
      return;
    }

    vec3 oColor = vec3(0);

    if (uPass == 0) {
      oColor += sampleHigherLevel(ivec2(-1, 0)) * 0.25;
      oColor += sampleHigherLevel(ivec2( 0, 0)) * 0.5;
      oColor += sampleHigherLevel(ivec2( 1, 0)) * 0.25;
    } else {
      oColor += sampleSameLevel(ivec2(0, -1)) * 0.25;
      oColor += sampleSameLevel(ivec2(0,  0)) * 0.5;
      oColor += sampleSameLevel(ivec2(0,  1)) * 0.25;
    }

    imageStore(uOutColor, storePos, vec4(oColor, 0.0));
  }
)";

////////////////////////////////////////////////////////////////////////////////////////////////////

GlowMipMap::GlowMipMap(int hdrBufferWidth, int hdrBufferHeight)
    : VistaTexture(GL_TEXTURE_2D)
    , mHDRBufferWidth(hdrBufferWidth)
    , mHDRBufferHeight(hdrBufferHeight)
    , mTemporaryTarget(new VistaTexture(GL_TEXTURE_2D)) {
  // create glow mipmap storage
  int iWidth  = mHDRBufferWidth / 2;
  int iHeight = mHDRBufferHeight / 2;
  mMaxLevels  = std::max(1.0, std::floor(std::log2(std::max(iWidth, iHeight))) + 1);

  Bind();
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

  glTexStorage2D(GL_TEXTURE_2D, mMaxLevels, GL_RGBA32F, iWidth, iHeight);

  // create storage for temporary glow target (this is used for the vertical blurring)
  mTemporaryTarget->Bind();
  glTexStorage2D(GL_TEXTURE_2D, mMaxLevels, GL_RGBA32F, iWidth, iHeight);

  // create compute shader
  auto        shader = glCreateShader(GL_COMPUTE_SHADER);
  const char* c_str  = sGlowShader.c_str();
  glShaderSource(shader, 1, &c_str, nullptr);
  glCompileShader(shader);

  auto val = 0;
  glGetShaderiv(shader, GL_COMPILE_STATUS, &val);
  if (val != GL_TRUE) {
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

  auto rvalue = 0;
  glGetProgramiv(mComputeProgram, GL_LINK_STATUS, &rvalue);
  if (!rvalue) {
    auto log_length = 0;
    glGetProgramiv(mComputeProgram, GL_INFO_LOG_LENGTH, &log_length);
    std::vector<char> v(log_length);
    glGetProgramInfoLog(mComputeProgram, log_length, nullptr, v.data());
    std::string log(begin(v), end(v));

    throw std::runtime_error(std::string("ERROR: Failed to link compute shader\n") + log);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

GlowMipMap::~GlowMipMap() {
  glDeleteProgram(mComputeProgram);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void GlowMipMap::update(VistaTexture* hdrBufferComposite) {
  int iWidth  = mHDRBufferWidth / 2;
  int iHeight = mHDRBufferHeight / 2;

  // update current glow mipmap ------------------------------------------------------------------

  glUseProgram(mComputeProgram);

  for (int level(0); level < mMaxLevels; ++level) {
    for (int pass(0); pass <= 1; ++pass) {
      VistaTexture *input = this, *output = this;
      int           inputLevel = level, outputLevel = level;

      // level  pass   input   inputLevel output outputLevel     blur      samplesHigherLevel
      //   0     0    gbuffer     0        temp      0        horizontal          true
      //   0     1     temp       0        this      0         vertical           false
      //   1     0     this       0        temp      1        horizontal          true
      //   1     1     temp       1        this      1         vertical           false
      //   2     0     this       1        temp      2        horizontal          true
      //   2     1     temp       2        this      2         vertical           false
      //   3     0     this       2        temp      3        horizontal          true
      //   3     1     temp       3        this      3         vertical           false

      if (pass == 0) {
        output     = mTemporaryTarget;
        inputLevel = std::max(0, inputLevel - 1);
      }

      if (pass == 1) {
        input = mTemporaryTarget;
      } else if (level == 0) {
        input = hdrBufferComposite;
      }

      glUniform1i(glGetUniformLocation(mComputeProgram, "uPass"), pass);

      int width  = std::max(1.0, std::floor(static_cast<double>(iWidth) / std::pow(2, level)));
      int height = std::max(1.0, std::floor(static_cast<double>(iHeight) / std::pow(2, level)));

      glBindImageTexture(0, output->GetId(), outputLevel, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F);
      glBindImageTexture(1, input->GetId(), inputLevel, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA32F);

      glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
      glDispatchCompute(std::ceil(1.0 * width / 16), std::ceil(1.0 * height / 16), 1);
    }
  }

  glBindImageTexture(0, 0, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F);
  glBindImageTexture(1, 0, 0, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA32F);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cs::graphics
