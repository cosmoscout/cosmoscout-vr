////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "GlareMipMap.hpp"

#include "logger.hpp"

#include <algorithm>
#include <cmath>
#include <glm/gtc/type_ptr.hpp>
#include <iostream>
#include <string>
#include <vector>

namespace cs::graphics {

////////////////////////////////////////////////////////////////////////////////////////////////////

static const char* sGlareShader = R"(
  layout (local_size_x = 16, local_size_y = 16) in;

  #if NUM_MULTISAMPLES > 0
    layout (rgba32f, binding = 0) readonly uniform image2DMS uInHDRBuffer;
  #else
    layout (rgba32f, binding = 0) readonly uniform image2D uInHDRBuffer;
  #endif

  layout (rgba32f, binding = 1) readonly  uniform image2D uInColor;
  layout (rgba32f, binding = 2) writeonly uniform image2D uOutColor;

  uniform int  uPass;
  uniform int  uLevel;
  uniform mat4 uMatP;
  uniform mat4 uMatInvP;

  // constants
  const float PI = 3.14159265359;

  vec3 sampleHDRBuffer(ivec2 pos) {
    #if NUM_MULTISAMPLES > 0
      // For performance reasons, we only use one sample for the glare.
      vec3 col = imageLoad(uInHDRBuffer, ivec2(pos*2 + ivec2(0,0)), 0).rgb * 0.25
               + imageLoad(uInHDRBuffer, ivec2(pos*2 + ivec2(1,0)), 0).rgb * 0.25
               + imageLoad(uInHDRBuffer, ivec2(pos*2 + ivec2(0,1)), 0).rgb * 0.25
               + imageLoad(uInHDRBuffer, ivec2(pos*2 + ivec2(1,1)), 0).rgb * 0.25;
    #else
      vec3 col = imageLoad(uInHDRBuffer, ivec2(pos*2 + ivec2(0,0))).rgb * 0.25
               + imageLoad(uInHDRBuffer, ivec2(pos*2 + ivec2(1,0))).rgb * 0.25
               + imageLoad(uInHDRBuffer, ivec2(pos*2 + ivec2(0,1))).rgb * 0.25
               + imageLoad(uInHDRBuffer, ivec2(pos*2 + ivec2(1,1))).rgb * 0.25;
    #endif
    return col;
  }

  vec3 sampleHDRBuffer(vec2 pos) {
    ivec2 ipos = ivec2(pos);
    vec3 tl = sampleHDRBuffer(ipos);
    vec3 tr = sampleHDRBuffer(ipos + ivec2(1, 0));
    vec3 bl = sampleHDRBuffer(ipos + ivec2(0, 1));
    vec3 br = sampleHDRBuffer(ipos + ivec2(1, 1));
    vec2 f  = fract(pos);
    vec3 tA = mix(tl, tr, f.x);
    vec3 tB = mix(bl, br, f.x);
    return mix(tA, tB, f.y);
  }



  vec3 sampleHigherLevel(ivec2 pos) {
    vec3 col = imageLoad(uInColor, ivec2(pos*2 + ivec2(0,0))).rgb * 0.25
             + imageLoad(uInColor, ivec2(pos*2 + ivec2(1,0))).rgb * 0.25
             + imageLoad(uInColor, ivec2(pos*2 + ivec2(0,1))).rgb * 0.25
             + imageLoad(uInColor, ivec2(pos*2 + ivec2(1,1))).rgb * 0.25;
    return col;
  }

  vec3 sampleHigherLevel(vec2 pos) {
    ivec2 ipos = ivec2(pos);
    vec3 tl = sampleHigherLevel(ipos);
    vec3 tr = sampleHigherLevel(ipos + ivec2(1, 0));
    vec3 bl = sampleHigherLevel(ipos + ivec2(0, 1));
    vec3 br = sampleHigherLevel(ipos + ivec2(1, 1));
    vec2 f  = fract(pos);
    vec3 tA = mix(tl, tr, f.x);
    vec3 tB = mix(bl, br, f.x);
    return mix(tA, tB, f.y);
  }


  vec3 sampleSameLevel(ivec2 pos) {
    return imageLoad(uInColor, pos).rgb;
  }

  vec3 sampleSameLevel(vec2 pos) {
    ivec2 ipos = ivec2(pos);
    vec3 tl = sampleSameLevel(ipos);
    vec3 tr = sampleSameLevel(ipos + ivec2(1, 0));
    vec3 bl = sampleSameLevel(ipos + ivec2(0, 1));
    vec3 br = sampleSameLevel(ipos + ivec2(1, 1));
    vec2 f  = fract(pos);
    vec3 tA = mix(tl, tr, f.x);
    vec3 tB = mix(bl, br, f.x);
    return mix(tA, tB, f.y);
  }

  mat3 rotationMatrix(vec3 axis, float angle) {
    float s = sin(angle);
    float c = cos(angle);
    float i = 1.0 - c;
    
    return mat3(i*axis.x*axis.x + c,        i*axis.x*axis.y - axis.z*s, i*axis.z*axis.x + axis.y*s,
                i*axis.x*axis.y + axis.z*s, i*axis.y*axis.y + c,        i*axis.y*axis.z - axis.x*s,
                i*axis.z*axis.x - axis.y*s, i*axis.y*axis.z + axis.x*s, i*axis.z*axis.z + c);
  }

  vec3 rotate(vec3 v, vec3 axis, float angle) {
    return rotationMatrix(axis, angle) * v;
  }

  float getGauss(float sigma, float value) {
    return 1.0 / (sigma*sqrt(2*PI)) * exp(-0.5*pow(value/sigma, 2));
  }

  void main() {
    ivec2 storePos = ivec2(gl_GlobalInvocationID.xy);
    ivec2 size     = imageSize(uOutColor);

    if (storePos.x >= size.x || storePos.y >= size.y) {
      return;
    }

    vec3 glare = vec3(0);

    #ifdef GLAREMODE_SYMMETRIC_GAUSS
      const float samples = 2*(GLARE_QUALITY+1)+1;
      const float sigma   = 1;
      float totalWeight   = 0;

      for (float i = 0; i < samples; ++i) {
        float offset = i - floor(samples/2);
        float weight = getGauss(sigma, offset);
        totalWeight += weight;

        if (uPass == 0 && uLevel == 0) {
          glare += sampleHDRBuffer(storePos+ivec2(offset, 0)) * weight;
        } else if (uPass == 0) {
          glare += sampleHigherLevel(storePos+ivec2(offset, 0)) * weight;
        } else {
          glare += sampleSameLevel(storePos+ivec2(0, offset)) * weight;
        }
      }

      glare /= totalWeight;
    #endif

    #ifdef GLAREMODE_ASYMMETRIC_GAUSS
      vec2 posClipSpace = 2.0 * vec2(gl_GlobalInvocationID.xy) / vec2(size) - 1.0;
      vec4 posViewSpace = uMatInvP * vec4(posClipSpace, 0.0, 1.0);

      vec3 rotAxisV = cross(posViewSpace.xyz, vec3(1, 0, 0));
      rotAxisV = cross(posViewSpace.xyz, rotAxisV);

      vec3 rotAxisD = cross(posViewSpace.xyz, vec3(0, 0, -1));
      if (posViewSpace.y < 0) {
        rotAxisD = -rotAxisD;
      }

      vec3 rotAxis = mix(rotAxisV, rotAxisD, clamp(2*abs(posViewSpace.y / length(posViewSpace.xyz)), 0, 1));

      if (uPass == 0) {
        rotAxis = normalize(rotAxis);
      } else {
        rotAxis = normalize(cross(rotAxis, posViewSpace.xyz));
      }

      const float samples    = 2*(GLARE_QUALITY+1)+1;
      const float totalAngle = pow(2, uLevel);
      float totalWeight      = 0;
        
      for (float i = 0; i < samples; ++i) {

        float angle  = totalAngle * i / (samples-1) - totalAngle * 0.5;

        float sigma = totalAngle / 4;
        float weight = getGauss(sigma, angle);

        vec4 pos = uMatP * vec4(rotate(posViewSpace.xyz, rotAxis, angle*PI/180.0), 1.0);
        pos /= pos.w;
        
        vec2 samplePos = (0.5*pos.xy + 0.5)*size;

        if (uPass == 0 && uLevel == 0) {
          glare += sampleHDRBuffer(samplePos) * weight;
        } else if (uPass == 0) {
          glare += sampleHigherLevel(samplePos) * weight;
        } else {
          glare += sampleSameLevel(samplePos) * weight;
        }

        totalWeight += weight;
      }

      glare /= totalWeight;

    #endif

    imageStore(uOutColor, storePos, vec4(glare, 0.0));
  }
)";

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
  glTexStorage2D(GL_TEXTURE_2D, mMaxLevels, GL_RGBA32F, iWidth, iHeight);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

GlareMipMap::~GlareMipMap() {
  glDeleteProgram(mComputeProgram);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void GlareMipMap::update(
    VistaTexture* hdrBufferComposite, HDRBuffer::GlareMode glareMode, uint32_t glareQuality) {

  if (mComputeProgram == 0 || glareMode != mLastGlareMode || glareQuality != mLastGlareQuality) {

    // Create the compute shader.
    auto        shader = glCreateShader(GL_COMPUTE_SHADER);
    std::string source = "#version 430\n";
    source += "#define NUM_MULTISAMPLES " + std::to_string(mHDRBufferSamples) + "\n";
    source += "#define GLARE_QUALITY " + std::to_string(glareQuality) + "\n";

    if (glareMode == HDRBuffer::GlareMode::eSymmetricGauss) {
      source += "#define GLAREMODE_SYMMETRIC_GAUSS\n";
    } else if (glareMode == HDRBuffer::GlareMode::eAsymmetricGauss) {
      source += "#define GLAREMODE_ASYMMETRIC_GAUSS\n";
    }

    source += sGlareShader;
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
      logger().error("ERROR: Failed to compile shader: {}", log);
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

      logger().error("ERROR: Failed to link compute shader: {}", log);
    }

    mUniforms.level                   = glGetUniformLocation(mComputeProgram, "uLevel");
    mUniforms.pass                    = glGetUniformLocation(mComputeProgram, "uPass");
    mUniforms.projectionMatrix        = glGetUniformLocation(mComputeProgram, "uMatP");
    mUniforms.inverseProjectionMatrix = glGetUniformLocation(mComputeProgram, "uMatInvP");

    mLastGlareMode    = glareMode;
    mLastGlareQuality = glareQuality;
  }

  // We update the glare mipmap with several passes. First, the base level is filled with a
  // downsampled and horizontally blurred version of the HDRBuffer. Then, this is blurred
  // vertically. Then it's downsampled and horizontally blurred once more. And so on.

  glUseProgram(mComputeProgram);

  glBindImageTexture(0, hdrBufferComposite->GetId(), 0, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA32F);

  int maxLevels = mMaxLevels;

  if (glareMode == HDRBuffer::GlareMode::eAsymmetricGauss) {
    std::array<GLfloat, 16> glMatP{};
    glGetFloatv(GL_PROJECTION_MATRIX, glMatP.data());
    glm::mat4 matP    = glm::make_mat4x4(glMatP.data());
    glm::mat4 matInvP = glm::inverse(matP);
    glUniformMatrix4fv(mUniforms.projectionMatrix, 1, GL_FALSE, glm::value_ptr(matP));
    glUniformMatrix4fv(mUniforms.inverseProjectionMatrix, 1, GL_FALSE, glm::value_ptr(matInvP));
  }

  for (int level(0); level < maxLevels; ++level) {
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
