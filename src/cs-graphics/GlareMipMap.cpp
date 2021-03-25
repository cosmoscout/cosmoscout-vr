////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "GlareMipMap.hpp"

#include "../cs-utils/FrameTimings.hpp"
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

  layout (rgba32f, binding = 1) readonly  uniform image2D uInGlare;
  layout (rgba32f, binding = 2) writeonly uniform image2D uOutGlare;

  uniform int  uPass;
  uniform int  uLevel;
  uniform mat4 uMatP;
  uniform mat4 uMatInvP;

  const float PI = 3.14159265359;

  // Makes four texture look-ups in the input HDRBuffer at the four pixels corresponding to
  // the pixel position in the base layer of the output mipmap pyramid.
  // For performance reasons, we only use one sample for multisample inputs.
  vec3 sampleHDRBuffer(ivec2 pos) {
    #if NUM_MULTISAMPLES > 0
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

  // Calls the method above four times in order to allow for bilinear interpolation. This is
  // required for floating point positions and results in sixteen texture look-ups.
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


  // Makes four texture look-ups in the input layer of the glare mipmap at the four pixels
  // corresponding to the pixel position in the current layer of the mipmap pyramid.
  vec3 sampleHigherLevel(ivec2 pos) {
    vec3 col = imageLoad(uInGlare, ivec2(pos*2 + ivec2(0,0))).rgb * 0.25
             + imageLoad(uInGlare, ivec2(pos*2 + ivec2(1,0))).rgb * 0.25
             + imageLoad(uInGlare, ivec2(pos*2 + ivec2(0,1))).rgb * 0.25
             + imageLoad(uInGlare, ivec2(pos*2 + ivec2(1,1))).rgb * 0.25;
    return col;
  }

  // Calls the method above four times in order to allow for bilinear interpolation. This is
  // required for floating point positions and results in sixteen texture look-ups.
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

  // Makes just one texture look-ups in the input layer of the glare mipmap at the given
  // pixel position.
  vec3 sampleSameLevel(ivec2 pos) {
    return imageLoad(uInGlare, pos).rgb;
  }

  // Calls the method above four times in order to allow for bilinear interpolation. This is
  // required for floating point positions and results in four texture look-ups.
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

  // Rotates the given vector around a given axis.
  // Based on comment from http://www.neilmendoza.com/glsl-rotation-about-an-arbitrary-axis/
  vec3 rotate(vec3 v, vec3 axis, float angle) {
    return mix(dot(axis, v) * axis, v, cos(angle)) + cross(axis, v) * sin(angle);
  }

  // Evalutes the normal distribution function for the given value.
  float getGauss(float sigma, float value) {
    return 1.0 / (sigma*sqrt(2*PI)) * exp(-0.5*pow(value/sigma, 2));
  }


  void main() {

    ivec2 pixelPos   = ivec2(gl_GlobalInvocationID.xy);
    ivec2 outputSize = imageSize(uOutGlare);

    // Discard any threads outside the output layer.
    if (pixelPos.x >= outputSize.x || pixelPos.y >= outputSize.y) {
      return;
    }

    // Take an odd number of samples to make sure that we sample the center value.
    const float samples = 2*(GLARE_QUALITY+1)+1;

    // These values will contain the accumulated glare values.
    vec3  glare       = vec3(0);
    float totalWeight = 0;

    // The first glare variant computes a symmetrical gaussian blur in screen space. It is separated
    // into a vertical and horizontal component which are computed in different passes.
    // If uPass == 0, horizontal blurring happens, if uPass == 1, vertical blurring happens.
    // This is not perspectively correct but very fast.

    #ifdef GLAREMODE_SYMMETRIC_GAUSS
      for (float i = 0; i < samples; ++i) {
        float offset = i - floor(samples/2);
        float weight = getGauss(1, offset);
        totalWeight += weight;

        // If we are writing to level zero, we have to sample the input HDR buffer. For all
        // successive levels we sample the previous level in the first passes and the same level
        // in the second passes.
        if (uPass == 0 && uLevel == 0) {
          glare += sampleHDRBuffer(pixelPos+ivec2(offset, 0)) * weight;
        } else if (uPass == 0) {
          glare += sampleHigherLevel(pixelPos+ivec2(offset, 0)) * weight;
        } else {
          glare += sampleSameLevel(pixelPos+ivec2(0, offset)) * weight;
        }
      }
    #endif


    // The second variant computes an asymmetric perspectively correct gaussian decomposed into two
    // components which can be roughly described as radial and circular. A naive implementation with
    // a strictly circular and a strictly radial component suffers from undefined behavior close to
    // the focal point (usually in the middle of the screen) resulting in weird glow patterns in
    // this area:
    //
    //             Primary Component            Secondary (orthogonal) Component
    //
    //                  \  |  /                              / --- \
    //                 --  X  --                            |   O   |
    //                  /  |  \                              \ --- /
    //                   radial                             circular
    //
    //
    // Therefore the implementation below uses two components like the following:
    //
    //                  \  |  /                               / -- \
    //                  |  |  |                              -- -- --
    //                  /  |  \                               \ -- /
    //            vertical / radial                  horizontal / circular

    #ifdef GLAREMODE_ASYMMETRIC_GAUSS

      // Reproject the current pixel position to view space.
      vec2 posClipSpace = 2.0 * vec2(gl_GlobalInvocationID.xy) / vec2(outputSize) - 1.0;
      vec4 posViewSpace = uMatInvP * vec4(posClipSpace, 0.0, 1.0);

      // The primary component depicted above is a mixture between a vertical rotation and a radial
      // rotation. A vertical rotation would require this axis:
      vec3 rotAxisVertical = cross(posViewSpace.xyz, vec3(1, 0, 0));
      rotAxisVertical = cross(posViewSpace.xyz, rotAxisVertical);

      // A radial rotation would be around this axis:
      vec3 rotAxisRadial = cross(posViewSpace.xyz, vec3(0, 0, -1));
      if (posViewSpace.y < 0) {
        rotAxisRadial = -rotAxisRadial;
      }

      // We mix those two axes with a factor which depends on the vertical position of our
      // pixel on the screen. The magic factor of two determines how fast we change from one to
      // another and is not very sensitive. A value of five would result in very similar images.
      float alpha = clamp(2*abs(posViewSpace.y / length(posViewSpace.xyz)), 0, 1);
      vec3 rotAxis = mix(rotAxisVertical, rotAxisRadial, alpha);

      // The primary passes use the axis computed above, the secondary passes use an
      // orthogonal rotation axis.
      if (uPass == 0) {
        rotAxis = normalize(rotAxis);
      } else {
        rotAxis = normalize(cross(rotAxis, posViewSpace.xyz));
      }

      // The angle covered by the gauss kernel increases quadratically with the mipmap level.
      const float totalAngle = pow(2, uLevel) * PI / 180.0;

      // Rotate the view vector to the current pixel several times around the rotation axis
      // in order to sample the vicinity.
      for (float i = 0; i < samples; ++i) {
        float angle  = totalAngle * i / (samples-1) - totalAngle * 0.5;
        float sigma  = totalAngle / MAX_LEVELS;
        float weight = getGauss(sigma, angle);

        // Compute the rotated sample position in screen space.
        vec4 pos = uMatP * vec4(rotate(posViewSpace.xyz, rotAxis, angle), 1.0);
        pos /= pos.w;
        
        // Convert to texture space.
        vec2 samplePos = (0.5*pos.xy + 0.5)*outputSize;

        // If we are writing to level zero, we have to sample the input HDR buffer. For all
        // successive levels we sample the previous level in the first passes and the same level
        // in the second passes.
        if (uPass == 0 && uLevel == 0) {
          glare += sampleHDRBuffer(samplePos) * weight;
        } else if (uPass == 0) {
          glare += sampleHigherLevel(samplePos) * weight;
        } else {
          glare += sampleSameLevel(samplePos) * weight;
        }

        totalWeight += weight;
      }
    #endif

    // Make sure that we do not add energy.
    glare /= totalWeight;

    // Finally store the glare value in the output layer of the glare mipmap.
    imageStore(uOutGlare, pixelPos, vec4(glare, 0.0));
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
  mTemporaryTarget->Unbind();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

GlareMipMap::~GlareMipMap() {
  glDeleteProgram(mComputeProgram);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void GlareMipMap::update(
    VistaTexture* hdrBufferComposite, HDRBuffer::GlareMode glareMode, uint32_t glareQuality) {

  utils::FrameTimings::ScopedTimer timer("Compute Glare");

  if (mComputeProgram == 0 || glareMode != mLastGlareMode || glareQuality != mLastGlareQuality) {

    // Create the compute shader.
    auto        shader = glCreateShader(GL_COMPUTE_SHADER);
    std::string source = "#version 430\n";
    source += "#define NUM_MULTISAMPLES " + std::to_string(mHDRBufferSamples) + "\n";
    source += "#define GLARE_QUALITY " + std::to_string(glareQuality) + "\n";
    source += "#define MAX_LEVELS " + std::to_string(mMaxLevels) + "\n";

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
  // In the asymmetric case, its not strictly horizontal and vertical - see the shader above for
  // details.

  glUseProgram(mComputeProgram);

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
  glBindImageTexture(2, 0, 0, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA32F);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cs::graphics
