////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "ToneMappingNode.hpp"

#include "HDRBuffer.hpp"

#include <VistaInterProcComm/Cluster/VistaClusterDataCollect.h>
#include <VistaInterProcComm/Cluster/VistaClusterDataSync.h>
#include <VistaKernel/Cluster/VistaClusterMode.h>
#include <VistaKernel/DisplayManager/VistaDisplayManager.h>
#include <VistaKernel/EventManager/VistaEventManager.h>
#include <VistaKernel/EventManager/VistaSystemEvent.h>
#include <VistaKernel/VistaFrameLoop.h>
#include <VistaKernel/VistaSystem.h>
#include <VistaMath/VistaBoundingBox.h>
#include <VistaOGLExt/VistaGLSLShader.h>
#include <VistaOGLExt/VistaTexture.h>

#include <glm/glm.hpp>
#include <limits>
#include <utility>

namespace cs::graphics {

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace internal {
// based on
// https://placeholderart.wordpress.com/2014/12/15/implementing-a-physically-based-camera-automatic-exposure/

// References:
// http://en.wikipedia.org/wiki/Film_speed
// http://en.wikipedia.org/wiki/Exposure_value
// http://en.wikipedia.org/wiki/Light_meter

// Notes:
// EV below refers to EV at ISO 100

const float MIN_ISO      = 100.F;
const float MAX_ISO      = 6400.F;
const float MIN_SHUTTER  = 1.F / 4000.F;
const float MAX_SHUTTER  = 1.F / 30.F;
const float MIN_APERTURE = 1.8F;

// Given an aperture, shutter speed, and exposure value compute the required ISO value
float ComputeISO(float aperture, float shutterSpeed, float ev) {
  return (std::pow(aperture, 2.F) * 100.0F) / (shutterSpeed * std::pow(2.0F, ev));
}

// Given the camera settings compute the current exposure value
float ComputeEV(float aperture, float shutterSpeed, float iso) {
  return std::log2((std::pow(aperture, 2.F) * 100.0F) / (shutterSpeed * iso));
}

// Using the light metering equation compute the target exposure value
float ComputeTargetEV(float luminance) {
  // K is a light meter calibration constant
  const float K = 12.5F;
  return std::log2(luminance * 100.0F / K);
}

void ApplyAperturePriority(
    float focalLength, float targetEV, float& aperture, float& shutterSpeed, float& iso) {
  // Start with the assumption that we want a shutter speed of 1/f
  shutterSpeed = 1.0F / (focalLength * 1000.0F);

  // Compute the resulting ISO if we left the shutter speed here
  iso = glm::clamp(ComputeISO(aperture, shutterSpeed, targetEV), MIN_ISO, MAX_ISO);

  // Figure out how far we were from the target exposure value
  float evDiff = targetEV - ComputeEV(aperture, shutterSpeed, iso);

  // Compute the final shutter speed
  shutterSpeed = glm::clamp(shutterSpeed * std::pow(2.0F, -evDiff), MIN_SHUTTER, MAX_SHUTTER);
}

void ApplyShutterPriority(
    float /*unused*/, float targetEV, float& aperture, float& shutterSpeed, float& iso) {
  // Start with the assumption that we want an aperture of 4.0
  aperture = 4.0F;

  // Compute the resulting ISO if we left the aperture here
  iso = glm::clamp(ComputeISO(aperture, shutterSpeed, targetEV), MIN_ISO, MAX_ISO);

  // Figure out how far we were from the target exposure value
  float evDiff = targetEV - ComputeEV(aperture, shutterSpeed, iso);

  // Compute the final aperture
  aperture = glm::clamp(aperture * std::pow(std::sqrt(2.0F), evDiff), MIN_APERTURE, MIN_APERTURE);
}

void ApplyProgramAuto(
    float focalLength, float targetEV, float& aperture, float& shutterSpeed, float& iso) {
  // Start with the assumption that we want an aperture of 4.0
  aperture = 4.0F;

  // Start with the assumption that we want a shutter speed of 1/f
  shutterSpeed = 1.0F / (focalLength * 1000.0F);

  // Compute the resulting ISO if we left both shutter and aperture here
  iso = glm::clamp(ComputeISO(aperture, shutterSpeed, targetEV), MIN_ISO, MAX_ISO);

  // Apply half the difference in EV to the aperture
  float evDiff = targetEV - ComputeEV(aperture, shutterSpeed, iso);
  aperture =
      glm::clamp(aperture * std::pow(std::sqrt(2.0F), evDiff * 0.5F), MIN_APERTURE, MIN_APERTURE);

  // Apply the remaining difference to the shutter speed
  evDiff       = targetEV - ComputeEV(aperture, shutterSpeed, iso);
  shutterSpeed = glm::clamp(shutterSpeed * std::pow(2.0F, -evDiff), MIN_SHUTTER, MAX_SHUTTER);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace internal

////////////////////////////////////////////////////////////////////////////////////////////////////

static const char* sVertexShader = R"(
  out vec2 vTexcoords;

  void main()
  {
    vTexcoords  = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
    gl_Position = vec4(vTexcoords * 2.0 - 1.0, 0.0, 1.0);
  }
)";

////////////////////////////////////////////////////////////////////////////////////////////////////

static const char* sFragmentShader = R"(
  in vec2 vTexcoords;

  layout(pixel_center_integer) in vec4 gl_FragCoord;

  #if NUM_MULTISAMPLES > 0
    layout (binding = 0) uniform sampler2DMS uComposite;
    layout (binding = 1) uniform sampler2DMS uDepth;
  #else
    layout (binding = 0) uniform sampler2D uComposite;
    layout (binding = 1) uniform sampler2D uDepth;
  #endif

  layout (binding = 2) uniform sampler2D uGlowMipMap;

  uniform float uExposure;
  uniform float uGlowIntensity;

  layout(location = 0) out vec3 oColor;

  // http://filmicworlds.com/blog/filmic-tonemapping-operators/
  float A = 0.15;
  float B = 0.50;
  float C = 0.10;
  float D = 0.20;
  float E = 0.02;
  float F = 0.30;
  float W = 11.2;

  vec3 Uncharted2Tonemap(vec3 x) {
    return ((x*(A*x+C*B)+D*E)/(x*(A*x+B)+D*F))-E/F;
  }
  
  float linear_to_srgb(float c) {
    const float a = 0.055;
    if(c <= 0.0031308)
      return 12.92*c;
    else
      return 1.055 * pow(c, 1.0/2.4) - 0.055;
  }

  vec3 linear_to_srgb(vec3 c) {
    return vec3(linear_to_srgb(c.r), linear_to_srgb(c.g), linear_to_srgb(c.b));
  }

  float w0(float a) {
    return (1.0 / 6.0) * (a * (a * (-a + 3.0) - 3.0) + 1.0);
  }

  float w1(float a) {
    return (1.0 / 6.0) * (a * a * (3.0 * a - 6.0) + 4.0);
  }

  float w2(float a) {
    return (1.0 / 6.0) * (a * (a * (-3.0 * a + 3.0) + 3.0) + 1.0);
  }

  float w3(float a) {
    return (1.0 / 6.0) * (a * a * a);
  }

  // g0 and g1 are the two amplitude functions
  float g0(float a) {
    return w0(a) + w1(a);
  }

  float g1(float a) {
    return w2(a) + w3(a);
  }

  // h0 and h1 are the two offset functions
  float h0(float a) {
    return -1.0 + w1(a) / (w0(a) + w1(a));
  }

  float h1(float a) {
    return 1.0 + w3(a) / (w2(a) + w3(a));
  }

  vec4 texture2D_bicubic(sampler2D tex, vec2 uv, int p_lod) {
    float lod = float(p_lod);
    vec2 tex_size = textureSize(uGlowMipMap, p_lod);
    vec2 pixel_size = 1.0 / tex_size;
    uv = uv * tex_size + 0.5;
    vec2 iuv = floor(uv);
    vec2 fuv = fract(uv);

    float g0x = g0(fuv.x);
    float g1x = g1(fuv.x);
    float h0x = h0(fuv.x);
    float h1x = h1(fuv.x);
    float h0y = h0(fuv.y);
    float h1y = h1(fuv.y);

    vec2 p0 = (vec2(iuv.x + h0x, iuv.y + h0y) - 0.5) * pixel_size;
    vec2 p1 = (vec2(iuv.x + h1x, iuv.y + h0y) - 0.5) * pixel_size;
    vec2 p2 = (vec2(iuv.x + h0x, iuv.y + h1y) - 0.5) * pixel_size;
    vec2 p3 = (vec2(iuv.x + h1x, iuv.y + h1y) - 0.5) * pixel_size;

    return (g0(fuv.y) * (g0x * textureLod(tex, p0, lod) + g1x * textureLod(tex, p1, lod))) +
            (g1(fuv.y) * (g0x * textureLod(tex, p2, lod) + g1x * textureLod(tex, p3, lod)));
  }

  void main() {
    #if NUM_MULTISAMPLES > 0
      vec3 color = vec3(0.0);
      for (int i = 0; i < NUM_MULTISAMPLES; ++i) {
        color += texelFetch(uComposite, ivec2(vTexcoords * textureSize(uComposite)), i).rgb;
      }
      color /= NUM_MULTISAMPLES;

      float depth = 1.0;
      for (int i = 0; i < NUM_MULTISAMPLES; ++i) {
        depth = min(depth, texelFetch(uDepth, ivec2(vTexcoords * textureSize(uDepth)), i).r);
      }
      gl_FragDepth = depth;
    #else
      vec3 color = texelFetch(uComposite, ivec2(vTexcoords * textureSize(uComposite, 0)), 0).rgb;
      gl_FragDepth = texelFetch(uDepth, ivec2(vTexcoords * textureSize(uDepth, 0)), 0).r;
    #endif

    vec3  glow = vec3(0);
    int maxLevels = textureQueryLevels(uGlowMipMap);
    float weight = 1;

    if (uGlowIntensity > 0) {
      for (int i=0; i<maxLevels; ++i) {
        glow += texture2D_bicubic(uGlowMipMap, vTexcoords, i).rgb / (i+1);
      }
      color = mix(color, glow / maxLevels, uGlowIntensity);
    }

    color = Uncharted2Tonemap(uExposure*color);

    vec3 whiteScale = vec3(1.0)/Uncharted2Tonemap(vec3(W));

    oColor = linear_to_srgb(color*whiteScale);
  }
)";

////////////////////////////////////////////////////////////////////////////////////////////////////

ToneMappingNode::ToneMappingNode(std::shared_ptr<HDRBuffer> hdrBuffer)
    : mHDRBuffer(std::move(hdrBuffer))
    , mShader(new VistaGLSLShader()) {

  std::string defines = "#version 430\n";
  defines += "#define NUM_MULTISAMPLES " + std::to_string(mHDRBuffer->getMultiSamples()) + "\n";

  mShader->InitVertexShaderFromString(defines + sVertexShader);
  mShader->InitFragmentShaderFromString(defines + sFragmentShader);
  mShader->Link();

  mUniforms.exposure      = mShader->GetUniformLocation("uExposure");
  mUniforms.glowIntensity = mShader->GetUniformLocation("uGlowIntensity");

  // Connect to the VSE_POSTGRAPHICS event. When this event is emitted, we will collect all
  // luminance values of the connected cluster nodes.
  VistaEventManager* pEventManager = GetVistaSystem()->GetEventManager();
  pEventManager->AddEventHandler(
      this, VistaSystemEvent::GetTypeId(), VistaSystemEvent::VSE_POSTGRAPHICS);

  mLuminanceCollect = GetVistaSystem()->GetClusterMode()->CreateDataCollect();
  mLuminanceSync    = GetVistaSystem()->GetClusterMode()->CreateDataSync();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

ToneMappingNode::~ToneMappingNode() {
  VistaEventManager* pEventManager = GetVistaSystem()->GetEventManager();
  pEventManager->RemEventHandler(
      this, VistaSystemEvent::GetTypeId(), VistaSystemEvent::VSE_POSTGRAPHICS);

  delete mLuminanceCollect;
  delete mLuminanceSync;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void ToneMappingNode::setExposure(float ev) {
  if (!mEnableAutoExposure) {
    mExposure = ev;
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

float ToneMappingNode::getExposure() const {
  return mExposure;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void ToneMappingNode::setExposureCompensation(float ev) {
  mExposureCompensation = ev;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

float ToneMappingNode::getExposureCompensation() const {
  return mExposureCompensation;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void ToneMappingNode::setMinAutoExposure(float ev) {
  mMinAutoExposure = ev;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

float ToneMappingNode::getMinAutoExposure() const {
  return mMinAutoExposure;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void ToneMappingNode::setMaxAutoExposure(float ev) {
  mMaxAutoExposure = ev;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

float ToneMappingNode::getMaxAutoExposure() const {
  return mMaxAutoExposure;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void ToneMappingNode::setExposureAdaptionSpeed(float speed) {
  mExposureAdaptionSpeed = speed;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

float ToneMappingNode::getExposureAdaptionSpeed() const {
  return mExposureAdaptionSpeed;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void ToneMappingNode::setEnableAutoExposure(bool value) {
  mEnableAutoExposure = value;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool ToneMappingNode::getEnableAutoExposure() const {
  return mEnableAutoExposure;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void ToneMappingNode::setGlowIntensity(float intensity) {
  mGlowIntensity = intensity;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

float ToneMappingNode::getGlowIntensity() const {
  return mGlowIntensity;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

float ToneMappingNode::getLastAverageLuminance() const {
  if (mGlobalLuminanceData.mPixelCount > 0 && mGlobalLuminanceData.mTotalLuminance > 0) {
    return mGlobalLuminanceData.mTotalLuminance /
           static_cast<float>(mGlobalLuminanceData.mPixelCount);
  }
  return 0;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

float ToneMappingNode::getLastMaximumLuminance() const {
  if (mGlobalLuminanceData.mMaximumLuminance > 0) {
    return mGlobalLuminanceData.mMaximumLuminance;
  }
  return 0;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool ToneMappingNode::ToneMappingNode::Do() {

  bool doCalculateExposure =
      GetVistaSystem()->GetDisplayManager()->GetCurrentRenderInfo()->m_eEyeRenderMode !=
      VistaDisplayManager::RenderInfo::ERM_RIGHT;

  if (doCalculateExposure && mEnableAutoExposure) {
    mHDRBuffer->calculateLuminance();

    // We accumulate all luminance values of this frame (can be multiple viewports and / or multiple
    // eyes). These values will be send to the master in VSE_POSTGRAPHICS and accumulated for all
    // clients. The result is stored in mGlobalLuminanceData and is in the next frame used for
    // exposure calculation
    auto size = mHDRBuffer->getCurrentViewPortSize();
    mLocalLuminanceData.mPixelCount += size.at(0) * size.at(1);
    mLocalLuminanceData.mTotalLuminance += mHDRBuffer->getTotalLuminance();
    mLocalLuminanceData.mMaximumLuminance += mHDRBuffer->getMaximumLuminance();

    // Calculate exposure based on last frame's average luminance Time-dependent visual adaptation
    // for fast realistic image display (https://dl.acm.org/citation.cfm?id=344810).
    if (mGlobalLuminanceData.mPixelCount > 0 && mGlobalLuminanceData.mTotalLuminance > 0) {
      auto  frameTime = static_cast<float>(GetVistaSystem()->GetFrameLoop()->GetAverageLoopTime());
      float averageLuminance = getLastAverageLuminance();
      mAutoExposure += (std::log2(1.F / averageLuminance) - mAutoExposure) *
                       (1.F - std::exp(-mExposureAdaptionSpeed * frameTime));
    }
  }

  if (mGlowIntensity > 0) {
    mHDRBuffer->updateGlowMipMap();
  }

  if (doCalculateExposure && mEnableAutoExposure) {
    mExposure = glm::clamp(mAutoExposure, mMinAutoExposure, mMaxAutoExposure);
  }

  float exposure = std::pow(2.F, mExposure + mExposureCompensation);

  mHDRBuffer->unbind();
  mHDRBuffer->getCurrentWriteAttachment()->Bind(GL_TEXTURE0);
  mHDRBuffer->getDepthAttachment()->Bind(GL_TEXTURE1);
  mHDRBuffer->getGlowMipMap()->Bind(GL_TEXTURE2);

  glPushAttrib(GL_ENABLE_BIT | GL_DEPTH_BUFFER_BIT);
  glDisable(GL_BLEND);
  glDisable(GL_CULL_FACE);
  glEnable(GL_TEXTURE_2D);

  mShader->Bind();
  mShader->SetUniform(mUniforms.exposure, exposure);
  mShader->SetUniform(mUniforms.glowIntensity, mGlowIntensity);

  glDrawArrays(GL_TRIANGLES, 0, 3);

  glPopAttrib();

  return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool ToneMappingNode::GetBoundingBox(VistaBoundingBox& oBoundingBox) {
  float      min(std::numeric_limits<float>::min());
  float      max(std::numeric_limits<float>::max());
  std::array fMin{min, min, min};
  std::array fMax{max, max, max};

  oBoundingBox.SetBounds(fMin.data(), fMax.data());

  return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void ToneMappingNode::HandleEvent(VistaEvent* pEvent) {
  // We accumulate all luminance values of this frame (can be multiple viewports and / or multiple
  // eyes) in mLocalLuminanceData. This will be send to the master in VSE_POSTGRAPHICS and
  // accumulated for all clients. The result is stored in mGlobalLuminanceData and is in the next
  // frame used for exposure calculation.
  if (pEvent->GetId() == VistaSystemEvent::VSE_POSTGRAPHICS) {
    std::vector<std::vector<VistaType::byte>> globalData;
    std::vector<VistaType::byte>              localData(sizeof(LuminanceData));

    std::memcpy(&localData[0], &mLocalLuminanceData, sizeof(LuminanceData));
    mLuminanceCollect->CollectData(&localData[0], sizeof(LuminanceData), globalData);

    // mGlobalLuminanceData is only filled on the cluster master. The slaves will receive the
    // accumulated mGlobalLuminanceData with the SyncData call below
    mGlobalLuminanceData.mPixelCount       = 0;
    mGlobalLuminanceData.mTotalLuminance   = 0;
    mGlobalLuminanceData.mMaximumLuminance = 0;

    for (auto const& data : globalData) {
      LuminanceData luminance;
      std::memcpy(&luminance, &data[0], sizeof(LuminanceData));
      mGlobalLuminanceData.mPixelCount += luminance.mPixelCount;
      mGlobalLuminanceData.mTotalLuminance += luminance.mTotalLuminance;
      mGlobalLuminanceData.mMaximumLuminance =
          std::max(mGlobalLuminanceData.mMaximumLuminance, luminance.mMaximumLuminance);
    }

    std::memcpy(&localData[0], &mGlobalLuminanceData, sizeof(LuminanceData));
    mLuminanceSync->SyncData(localData);
    std::memcpy(&mGlobalLuminanceData, &localData[0], sizeof(LuminanceData));

    // Reset local data for next frame.
    mLocalLuminanceData.mPixelCount       = 0;
    mLocalLuminanceData.mTotalLuminance   = 0;
    mLocalLuminanceData.mMaximumLuminance = 0;
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cs::graphics
