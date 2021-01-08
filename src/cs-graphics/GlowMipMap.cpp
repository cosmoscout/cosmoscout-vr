////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "GlowMipMap.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

namespace cs::graphics {

////////////////////////////////////////////////////////////////////////////////////////////////////

static const char* sGlowShader = R"(
  layout (local_size_x = 16, local_size_y = 16) in;

  #if NUM_MULTISAMPLES > 0
    layout (rgba32f, binding = 0) readonly uniform image2DMS uInHDRBuffer;
  #else
    layout (rgba32f, binding = 0) readonly uniform image2D uInHDRBuffer;
  #endif

  layout (rgba32f, binding = 1) readonly  uniform image2D uInColor;
  layout (rgba32f, binding = 2) writeonly uniform image2D uOutColor;

  uniform int uPass;
  uniform int uLevel;

  vec3 sampleHDRBuffer(ivec2 offset) {
    #if NUM_MULTISAMPLES > 0
      // For performance reasons, we only use one sample for the glow.
      vec3 col = imageLoad(uInHDRBuffer, ivec2((gl_GlobalInvocationID.xy+offset)*2 + ivec2(0,0)), 0).rgb * 0.25
               + imageLoad(uInHDRBuffer, ivec2((gl_GlobalInvocationID.xy+offset)*2 + ivec2(1,0)), 0).rgb * 0.25
               + imageLoad(uInHDRBuffer, ivec2((gl_GlobalInvocationID.xy+offset)*2 + ivec2(0,1)), 0).rgb * 0.25
               + imageLoad(uInHDRBuffer, ivec2((gl_GlobalInvocationID.xy+offset)*2 + ivec2(1,1)), 0).rgb * 0.25;
    #else
      vec3 col = imageLoad(uInHDRBuffer, ivec2((gl_GlobalInvocationID.xy+offset)*2 + ivec2(0,0))).rgb * 0.25
               + imageLoad(uInHDRBuffer, ivec2((gl_GlobalInvocationID.xy+offset)*2 + ivec2(1,0))).rgb * 0.25
               + imageLoad(uInHDRBuffer, ivec2((gl_GlobalInvocationID.xy+offset)*2 + ivec2(0,1))).rgb * 0.25
               + imageLoad(uInHDRBuffer, ivec2((gl_GlobalInvocationID.xy+offset)*2 + ivec2(1,1))).rgb * 0.25;
    #endif
    return col;
  }

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

    if (uPass == 0 && uLevel == 0) {
      oColor += sampleHDRBuffer(ivec2(-1, 0)) * 0.25;
      oColor += sampleHDRBuffer(ivec2( 0, 0)) * 0.5;
      oColor += sampleHDRBuffer(ivec2( 1, 0)) * 0.25;
    } else if (uPass == 0) {
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

GlowMipMap::GlowMipMap(uint32_t hdrBufferSamples, int hdrBufferWidth, int hdrBufferHeight)
    : VistaTexture(GL_TEXTURE_2D)
    , mHDRBufferSamples(hdrBufferSamples)
    , mHDRBufferWidth(hdrBufferWidth)
    , mHDRBufferHeight(hdrBufferHeight)
    , mTemporaryTarget(new VistaTexture(GL_TEXTURE_2D)) {

  // Create glow mipmap storage. The texture has half the size of the HDR buffer (rounded down) in
  // both directions.
  int iWidth  = mHDRBufferWidth / 2;
  int iHeight = mHDRBufferHeight / 2;

  // Compute the number of available mipmap levels.
  mMaxLevels =
      static_cast<int>(std::max(1.0, std::floor(std::log2(std::max(iWidth, iHeight))) + 1));

  Bind();
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

  glTexStorage2D(GL_TEXTURE_2D, mMaxLevels, GL_RGBA32F, iWidth, iHeight);

  // Create storage for temporary glow target (this is used for the vertical blurring passes).
  mTemporaryTarget->Bind();
  glTexStorage2D(GL_TEXTURE_2D, mMaxLevels, GL_RGBA32F, iWidth, iHeight);

  // Create the compute shader.
  auto        shader = glCreateShader(GL_COMPUTE_SHADER);
  std::string source = "#version 430\n";
  source += "#define NUM_MULTISAMPLES " + std::to_string(mHDRBufferSamples) + "\n";
  source += sGlowShader;
  const char* pSource = source.c_str();
  glShaderSource(shader, 1, &pSource, nullptr);
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

  mUniforms.level = glGetUniformLocation(mComputeProgram, "uLevel");
  mUniforms.pass  = glGetUniformLocation(mComputeProgram, "uPass");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

GlowMipMap::~GlowMipMap() {
  glDeleteProgram(mComputeProgram);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void GlowMipMap::update(VistaTexture* hdrBufferComposite) {

  // We update the glow mipmap with several passes. First, the base level is filled with a
  // downsampled and horizontally blurred version of the HDRBuffer. Then, this is blurred
  // vertically. Then it's downsampled and horizontally blurred once more. And so on.

  glUseProgram(mComputeProgram);

  glBindImageTexture(0, hdrBufferComposite->GetId(), 0, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA32F);

  for (int level(0); level < mMaxLevels; ++level) {
    glUniform1i(mUniforms.level, level);

    for (int pass(0); pass < 2; ++pass) {
      VistaTexture* input       = this;
      VistaTexture* output      = this;
      int           inputLevel  = level;
      int           outputLevel = level;

      // level  pass   input   inputLevel output outputLevel     blur      samplesHigherLevel
      //   0     0   hdrbuffer    0        temp      0        horizontal          true
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
      }

      glUniform1i(mUniforms.pass, pass);

      int width = static_cast<int>(
          std::max(1.0, std::floor(static_cast<double>(static_cast<int>(mHDRBufferWidth / 2)) /
                                   std::pow(2, level))));
      int height = static_cast<int>(
          std::max(1.0, std::floor(static_cast<double>(static_cast<int>(mHDRBufferHeight / 2)) /
                                   std::pow(2, level))));

      glBindImageTexture(1, input->GetId(), inputLevel, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA32F);
      glBindImageTexture(2, output->GetId(), outputLevel, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F);

      glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
      glDispatchCompute(static_cast<uint32_t>(std::ceil(1.0 * width / 16)),
          static_cast<uint32_t>(std::ceil(1.0 * height / 16)), 1);
    }
  }

  glBindImageTexture(0, 0, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F);
  glBindImageTexture(1, 0, 0, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA32F);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cs::graphics
