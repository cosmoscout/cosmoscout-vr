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
#include <VistaKernel/EventManager/VistaEventManager.h>
#include <VistaKernel/EventManager/VistaSystemEvent.h>
#include <VistaKernel/VistaFrameLoop.h>
#include <VistaKernel/VistaSystem.h>
#include <VistaMath/VistaBoundingBox.h>
#include <VistaOGLExt/VistaGLSLShader.h>
#include <VistaOGLExt/VistaTexture.h>

#include <glm/glm.hpp>
#include <limits>

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

const float MIN_ISO      = 100;
const float MAX_ISO      = 6400;
const float MIN_SHUTTER  = 1.f / 4000.f;
const float MAX_SHUTTER  = 1.f / 30.f;
const float MIN_APERTURE = 1.8f;
const float MAX_APERTURE = 22.f;

// Given an aperture, shutter speed, and exposure value compute the required ISO value
float ComputeISO(float aperture, float shutterSpeed, float ev) {
  return (std::pow(aperture, 2.f) * 100.0f) / (shutterSpeed * std::pow(2.0f, ev));
}

// Given the camera settings compute the current exposure value
float ComputeEV(float aperture, float shutterSpeed, float iso) {
  return std::log2((std::pow(aperture, 2.f) * 100.0f) / (shutterSpeed * iso));
}

// Using the light metering equation compute the target exposure value
float ComputeTargetEV(float luminance) {
  // K is a light meter calibration constant
  const float K = 12.5f;
  return std::log2(luminance * 100.0f / K);
}

void ApplyAperturePriority(
    float focalLength, float targetEV, float& aperture, float& shutterSpeed, float& iso) {
  // Start with the assumption that we want a shutter speed of 1/f
  shutterSpeed = 1.0f / (focalLength * 1000.0f);

  // Compute the resulting ISO if we left the shutter speed here
  iso = glm::clamp(ComputeISO(aperture, shutterSpeed, targetEV), MIN_ISO, MAX_ISO);

  // Figure out how far we were from the target exposure value
  float evDiff = targetEV - ComputeEV(aperture, shutterSpeed, iso);

  // Compute the final shutter speed
  shutterSpeed = glm::clamp(shutterSpeed * std::pow(2.0f, -evDiff), MIN_SHUTTER, MAX_SHUTTER);
}

void ApplyShutterPriority(
    float focalLength, float targetEV, float& aperture, float& shutterSpeed, float& iso) {
  // Start with the assumption that we want an aperture of 4.0
  aperture = 4.0f;

  // Compute the resulting ISO if we left the aperture here
  iso = glm::clamp(ComputeISO(aperture, shutterSpeed, targetEV), MIN_ISO, MAX_ISO);

  // Figure out how far we were from the target exposure value
  float evDiff = targetEV - ComputeEV(aperture, shutterSpeed, iso);

  // Compute the final aperture
  aperture = glm::clamp(aperture * std::pow(std::sqrt(2.0f), evDiff), MIN_APERTURE, MIN_APERTURE);
}

void ApplyProgramAuto(
    float focalLength, float targetEV, float& aperture, float& shutterSpeed, float& iso) {
  // Start with the assumption that we want an aperture of 4.0
  aperture = 4.0f;

  // Start with the assumption that we want a shutter speed of 1/f
  shutterSpeed = 1.0f / (focalLength * 1000.0f);

  // Compute the resulting ISO if we left both shutter and aperture here
  iso = glm::clamp(ComputeISO(aperture, shutterSpeed, targetEV), MIN_ISO, MAX_ISO);

  // Apply half the difference in EV to the aperture
  float evDiff = targetEV - ComputeEV(aperture, shutterSpeed, iso);
  aperture =
      glm::clamp(aperture * std::pow(std::sqrt(2.0f), evDiff * 0.5f), MIN_APERTURE, MIN_APERTURE);

  // Apply the remaining difference to the shutter speed
  evDiff       = targetEV - ComputeEV(aperture, shutterSpeed, iso);
  shutterSpeed = glm::clamp(shutterSpeed * std::pow(2.0f, -evDiff), MIN_SHUTTER, MAX_SHUTTER);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace internal

////////////////////////////////////////////////////////////////////////////////////////////////////

const std::string ToneMappingNode::sVertexShader = R"(
  #version 430 compatibility

  out vec2 vTexcoords;

  void main()
  {
    vTexcoords  = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
    gl_Position = vec4(vTexcoords * 2.0 - 1.0, 0.0, 1.0);
  }
)";

////////////////////////////////////////////////////////////////////////////////////////////////////

const std::string ToneMappingNode::sFragmentShader = R"(
  #version 430 compatibility
  
  in vec2 vTexcoords;

  layout(pixel_center_integer) in vec4 gl_FragCoord;

  layout(binding = 0) uniform sampler2D uDepth;
  layout(binding = 1) uniform sampler2D uComposite;
  layout(binding = 2) uniform sampler2D uGlowMipMap;

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
    vec3 color = texelFetch(uComposite, ivec2(vTexcoords * textureSize(uComposite, 0)), 0).rgb;

    vec3  glow = vec3(0);
    int maxLevels = textureQueryLevels(uGlowMipMap);
    float weight = 1;

    if (uGlowIntensity > 0) {
      for (int i=0; i<maxLevels; ++i) {
        glow += texture2D_bicubic(uGlowMipMap, vTexcoords, i).rgb / (i+1);
      }
      color = mix(color, glow / maxLevels, uGlowIntensity);
    }

    gl_FragDepth = texelFetch(uDepth, ivec2(vTexcoords * textureSize(uDepth, 0)), 0).r;
    
    color = Uncharted2Tonemap(uExposure*color);

    vec3 whiteScale = vec3(1.0)/Uncharted2Tonemap(vec3(W));

    oColor = linear_to_srgb(color*whiteScale);
  }
)";

////////////////////////////////////////////////////////////////////////////////////////////////////

ToneMappingNode::ToneMappingNode(std::shared_ptr<HDRBuffer> const& hdrBuffer, bool drawToBackBuffer)
    : mHDRBuffer(hdrBuffer)
    , mDrawToBackBuffer(drawToBackBuffer)
    , mShader(new VistaGLSLShader()) {
  mShader->InitVertexShaderFromString(sVertexShader);
  mShader->InitFragmentShaderFromString(sFragmentShader);
  mShader->Link();

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

void ToneMappingNode::setExposureMeteringMode(ExposureMeteringMode value) {
  mExposureMeteringMode = value;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

ExposureMeteringMode ToneMappingNode::getExposureMeteringMode() const {
  return mExposureMeteringMode;
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
  if (mGlobalLuminaceData.mPixelCount > 0 && mGlobalLuminaceData.mTotalLuminance > 0) {
    return mGlobalLuminaceData.mTotalLuminance / mGlobalLuminaceData.mPixelCount;
  }
  return 0;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

float ToneMappingNode::getLastMaximumLuminance() const {
  if (mGlobalLuminaceData.mMaximumLuminance > 0) {
    return mGlobalLuminaceData.mMaximumLuminance;
  }
  return 0;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool ToneMappingNode::ToneMappingNode::Do() {
  if (mEnableAutoExposure) {
    mHDRBuffer->calculateLuminance(mExposureMeteringMode);

    // we accumulate all luminance values of this frame (can be multiple viewports
    // and / or multiple eyes). These values will be send to the master in
    // VSE_POSTGRAPHICS and accumulated for all clients. The result is stored in
    // mGlobalLuminaceData and is in the next frame used for exposure calculation
    auto size = mHDRBuffer->getCurrentViewPortSize();
    mLocalLuminaceData.mPixelCount += size[0] * size[1];
    mLocalLuminaceData.mTotalLuminance += mHDRBuffer->getTotalLuminance();
    mLocalLuminaceData.mMaximumLuminance += mHDRBuffer->getMaximumLuminance();

    // calculate exposure based on last frame's average luminance
    // Time-dependent visual adaptation for fast realistic image display
    // https://dl.acm.org/citation.cfm?id=344810
    if (mGlobalLuminaceData.mPixelCount > 0 && mGlobalLuminaceData.mTotalLuminance > 0) {
      float frameTime        = GetVistaSystem()->GetFrameLoop()->GetAverageLoopTime();
      float averageLuminance = getLastAverageLuminance();
      mAutoExposure += (std::log2(1.f / averageLuminance) - mAutoExposure) *
                       (1.f - std::exp(-mExposureAdaptionSpeed * frameTime));
    }
  }

  if (mGlowIntensity > 0) {
    mHDRBuffer->updateGlowMipMap();
  }
  // ---------------

  // float targetEV    = internal::ComputeTargetEV(mLuminance);
  // float focalLength = 25.f;
  // float aperture, shutterSpeed, iso;
  // internal::ApplyProgramAuto(focalLength, targetEV, aperture, shutterSpeed, iso);

  // std::cout << "--------------------" << std::endl;
  // std::cout << "targetEV:     " << targetEV << std::endl;
  // std::cout << "aperture:     " << aperture << std::endl;
  // std::cout << "shutterSpeed: 1 / " << 1.f / shutterSpeed << std::endl;
  // std::cout << "iso:          " << iso << std::endl;

  // ---------------

  if (mEnableAutoExposure) {
    mExposure = glm::clamp(mAutoExposure, mMinAutoExposure, mMaxAutoExposure);
  }

  float exposure = std::pow(2.f, mExposure + mExposureCompensation);

  mHDRBuffer->unbind();
  mHDRBuffer->getDepthAttachment()->Bind(GL_TEXTURE0);
  mHDRBuffer->getCurrentWriteAttachment()->Bind(GL_TEXTURE1);
  mHDRBuffer->getGlowMipMap()->Bind(GL_TEXTURE2);

  glPushAttrib(GL_ENABLE_BIT | GL_DEPTH_BUFFER_BIT);
  glDisable(GL_BLEND);
  glDisable(GL_CULL_FACE);
  glEnable(GL_TEXTURE_2D);

  // if we are drawing to the back-buffer, we want to
  // disable depth testing but enable depth writing
  // if we draw to the gbuffer we do not want to perform
  // neither depth testing nor depth writing
  if (mDrawToBackBuffer) {
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_ALWAYS);
    glDepthMask(GL_TRUE);
  } else {
    glDisable(GL_DEPTH_TEST);
    glDepthMask(GL_FALSE);
  }

  mShader->Bind();
  mShader->SetUniform(mShader->GetUniformLocation("uExposure"), exposure);
  mShader->SetUniform(mShader->GetUniformLocation("uGlowIntensity"), mGlowIntensity);

  glDrawArrays(GL_TRIANGLES, 0, 3);

  glPopAttrib();

  return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool ToneMappingNode::GetBoundingBox(VistaBoundingBox& oBoundingBox) {
  float min(std::numeric_limits<float>::min());
  float max(std::numeric_limits<float>::max());
  float fMin[3] = {min, min, min};
  float fMax[3] = {max, max, max};

  oBoundingBox.SetBounds(fMin, fMax);

  return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void ToneMappingNode::HandleEvent(VistaEvent* pEvent) {
  // we accumulate all luminance values of this frame (can be multiple viewports
  // and / or multiple eyes) in mLocalLuminaceData. This will be send to the master in
  // VSE_POSTGRAPHICS and accumulated for all clients. The result is stored in
  // mGlobalLuminaceData and is in the next frame used for exposure calculation
  if (pEvent->GetId() == VistaSystemEvent::VSE_POSTGRAPHICS) {
    std::vector<std::vector<VistaType::byte>> globalData;
    std::vector<VistaType::byte>              localData(sizeof(LuminanceData));

    std::memcpy(&localData[0], &mLocalLuminaceData, sizeof(LuminanceData));
    mLuminanceCollect->CollectData(&localData[0], sizeof(LuminanceData), globalData);

    // globalData is only filled on the cluster master. The slaves will receive
    // the accumulated mGlobalLuminaceData with the SyncData call below
    mGlobalLuminaceData.mPixelCount       = 0;
    mGlobalLuminaceData.mTotalLuminance   = 0;
    mGlobalLuminaceData.mMaximumLuminance = 0;

    for (auto const& data : globalData) {
      LuminanceData luminance;
      std::memcpy(&luminance, &data[0], sizeof(LuminanceData));
      mGlobalLuminaceData.mPixelCount += luminance.mPixelCount;
      mGlobalLuminaceData.mTotalLuminance += luminance.mTotalLuminance;
      mGlobalLuminaceData.mMaximumLuminance =
          std::max(mGlobalLuminaceData.mMaximumLuminance, luminance.mMaximumLuminance);
    }

    std::memcpy(&localData[0], &mGlobalLuminaceData, sizeof(LuminanceData));
    mLuminanceSync->SyncData(localData);
    std::memcpy(&mGlobalLuminaceData, &localData[0], sizeof(LuminanceData));

    // reset local data for next frame
    mLocalLuminaceData.mPixelCount       = 0;
    mLocalLuminaceData.mTotalLuminance   = 0;
    mLocalLuminaceData.mMaximumLuminance = 0;
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cs::graphics
