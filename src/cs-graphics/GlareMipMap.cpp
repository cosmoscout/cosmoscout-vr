////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "GlareMipMap.hpp"

#include "../cs-utils/FrameStats.hpp"
#include "../cs-utils/filesystem.hpp"
#include "logger.hpp"

#include <algorithm>
#include <cmath>
#include <glm/gtc/type_ptr.hpp>
#include <iostream>
#include <string>
#include <vector>

namespace cs::graphics {

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace {
GLint createComputeShader(std::string const& source) {
  auto shader  = glCreateShader(GL_COMPUTE_SHADER);
  auto pSource = source.c_str();
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
    logger().error("ERROR: Failed to compile shader: {}", log);
  }

  auto program = glCreateProgram();
  glAttachShader(program, shader);
  glLinkProgram(program);
  glDeleteShader(shader);

  glGetProgramiv(program, GL_LINK_STATUS, &val);
  if (!val) {
    auto log_length = 0;
    glGetProgramiv(program, GL_INFO_LOG_LENGTH, &log_length);
    std::vector<char> v(log_length);
    glGetProgramInfoLog(program, log_length, nullptr, v.data());
    std::string log(begin(v), end(v));
    logger().error("ERROR: Failed to link compute shader: {}", log);
  }

  return program;
}
} // namespace

////////////////////////////////////////////////////////////////////////////////////////////////////

GlareMipMap::GlareMipMap(uint32_t hdrBufferSamples, int hdrBufferWidth, int hdrBufferHeight)
    : VistaTexture(GL_TEXTURE_2D)
    , mHDRBufferSamples(hdrBufferSamples)
    , mHDRBufferWidth(hdrBufferWidth)
    , mHDRBufferHeight(hdrBufferHeight)
    , mTemporaryTarget(new VistaTexture(GL_TEXTURE_2D)) {

  // Create glare mipmap storage. The texture has half the size of the HDR buffer (rounded down) in
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

  // Create storage for temporary glare target (this is used for the vertical blurring passes).
  mTemporaryTarget->Bind();
  mTemporaryTarget->SetMinFilter(GL_LINEAR_MIPMAP_LINEAR);
  mTemporaryTarget->SetMagFilter(GL_LINEAR);
  mTemporaryTarget->SetWrapS(GL_CLAMP_TO_EDGE);
  mTemporaryTarget->SetWrapT(GL_CLAMP_TO_EDGE);

  glTexStorage2D(GL_TEXTURE_2D, mMaxLevels, GL_RGBA32F, iWidth, iHeight);
  mTemporaryTarget->Unbind();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

GlareMipMap::~GlareMipMap() {
  glDeleteProgram(mGlareProgram);
  glDeleteProgram(mCompositeProgram);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void GlareMipMap::update(
    VistaTexture* hdrBufferComposite, HDRBuffer::GlareMode glareMode, uint32_t glareQuality, float glareIntensity) {

  utils::FrameStats::ScopedTimer timer("Compute Glare");

  if (mGlareProgram == 0 || glareMode != mLastGlareMode || glareQuality != mLastGlareQuality || glareIntensity != mLastGlareIntensity) {

    // Create the compute shader.
    std::string source = "#version 430\n";
    source += "#define NUM_MULTISAMPLES " + std::to_string(mHDRBufferSamples) + "\n";
    source += "#define GLARE_QUALITY " + std::to_string(glareQuality) + "\n";
    source += "#define GLARE_INTENSITY " + std::to_string(glareIntensity) + "\n";
    source += "#define MAX_LEVELS " + std::to_string(mMaxLevels) + "\n";

    if (glareMode == HDRBuffer::GlareMode::eSymmetricGauss) {
      source += "#define GLAREMODE_SYMMETRIC_GAUSS\n";
    } else if (glareMode == HDRBuffer::GlareMode::eAsymmetricGauss) {
      source += "#define GLAREMODE_ASYMMETRIC_GAUSS\n";
    }

    std::string glareMipMapComputeShaderSource = source;
    glareMipMapComputeShaderSource += utils::filesystem::loadToString("../share/resources/shaders/glare.comp");

    mGlareProgram = createComputeShader(glareMipMapComputeShaderSource);

    mUniforms.level                   = glGetUniformLocation(mGlareProgram, "uLevel");
    mUniforms.pass                    = glGetUniformLocation(mGlareProgram, "uPass");
    mUniforms.projectionMatrix        = glGetUniformLocation(mGlareProgram, "uMatP");
    mUniforms.inverseProjectionMatrix = glGetUniformLocation(mGlareProgram, "uMatInvP");


    std::string glareCompositeComputeShaderSource = source;
    glareCompositeComputeShaderSource += utils::filesystem::loadToString("../share/resources/shaders/glareComposite.comp");
    mCompositeProgram = createComputeShader(glareCompositeComputeShaderSource);

    mLastGlareMode    = glareMode;
    mLastGlareQuality = glareQuality;
    mLastGlareIntensity = glareIntensity;
  }

  // We update the glare mipmap with several passes. First, the base level is filled with a
  // downsampled and horizontally blurred version of the HDRBuffer. Then, this is blurred
  // vertically. Then it's downsampled and horizontally blurred once more. And so on.
  // In the asymmetric case, its not strictly horizontal and vertical - see the shader above for
  // details.

  glUseProgram(mGlareProgram);

  glBindImageTexture(0, hdrBufferComposite->GetId(), 0, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA32F);

  // The asymmetric variant requires the projection and the inverse projection matrices.
  if (glareMode == HDRBuffer::GlareMode::eAsymmetricGauss) {
    std::array<GLfloat, 16> glMatP{};
    glGetFloatv(GL_PROJECTION_MATRIX, glMatP.data());
    glm::mat4 matP    = glm::make_mat4x4(glMatP.data());
    glm::mat4 matInvP = glm::inverse(matP);
    glUniformMatrix4fv(mUniforms.projectionMatrix, 1, GL_FALSE, glm::value_ptr(matP));
    glUniformMatrix4fv(mUniforms.inverseProjectionMatrix, 1, GL_FALSE, glm::value_ptr(matInvP));
  }

  for (int level(0); level < mMaxLevels; ++level) {
    glUniform1i(mUniforms.level, level);

    for (int pass(0); pass < 2; ++pass) {
      VistaTexture* input       = this;
      VistaTexture* output      = this;
      int           inputLevel  = level;
      int           outputLevel = level;

      // level  pass   input   inputLevel output outputLevel     blur      samplesHigherLevel
      //   0     0   hdrbuffer    0        this      0        horizontal          true
      //   0     1     this       0        temp      0         vertical           false
      //   1     0     temp       0        this      1        horizontal          true
      //   1     1     this       1        temp      1         vertical           false
      //   2     0     temp       1        this      2        horizontal          true
      //   2     1     this       2        temp      2         vertical           false
      //   3     0     temp       2        this      3        horizontal          true
      //   3     1     this       3        temp      3         vertical           false

      if (pass == 0) {
        input      = mTemporaryTarget;
        inputLevel = std::max(0, inputLevel - 1);
      }

      if (pass == 1) {
        output = mTemporaryTarget;
      }

      glUniform1i(mUniforms.pass, pass);

      int width = static_cast<int>(
          std::max(1.0, std::floor(static_cast<double>(static_cast<int>(mHDRBufferWidth / 2)) /
                                   std::pow(2, level))));
      int height = static_cast<int>(
          std::max(1.0, std::floor(static_cast<double>(static_cast<int>(mHDRBufferHeight / 2)) /
                                   std::pow(2, level))));

      input->Bind(GL_TEXTURE0);
      glBindImageTexture(1, input->GetId(), inputLevel, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA32F);
      glBindImageTexture(2, output->GetId(), outputLevel, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F);

      glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
      glDispatchCompute(static_cast<uint32_t>(std::ceil(1.0 * width / 16)),
          static_cast<uint32_t>(std::ceil(1.0 * height / 16)), 1);
    }
  }
  glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

  glUseProgram(mCompositeProgram);
  mTemporaryTarget->Bind(GL_TEXTURE0);
  glBindImageTexture(2, this->GetId(), 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F);
  glDispatchCompute(static_cast<uint32_t>(std::ceil(0.5 * mHDRBufferWidth / 16)),
      static_cast<uint32_t>(std::ceil(0.5 * mHDRBufferHeight / 16)), 1);

  glBindImageTexture(0, 0, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F);
  glBindImageTexture(1, 0, 0, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA32F);
  glBindImageTexture(2, 0, 0, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA32F);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cs::graphics
