////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "ToneMappingNode.hpp"

#include "../cs-utils/FrameStats.hpp"
#include "../cs-utils/filesystem.hpp"
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

ToneMappingNode::ToneMappingNode(std::shared_ptr<HDRBuffer> hdrBuffer)
    : mHDRBuffer(std::move(hdrBuffer)) {

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

void ToneMappingNode::setGlareIntensity(float intensity) {
  mGlareIntensity = intensity;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

float ToneMappingNode::getGlareIntensity() const {
  return mGlareIntensity;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void ToneMappingNode::setToneMappingMode(ToneMappingNode::ToneMappingMode mode) {
  if (mToneMappingMode != mode) {
    mToneMappingMode = mode;
    mShaderDirty     = true;
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

ToneMappingNode::ToneMappingMode ToneMappingNode::getToneMappingMode() const {
  return mToneMappingMode;
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

  utils::FrameStats::ScopedTimer timer("Tonemapping");

  if (mShaderDirty) {

    std::string defines = "#version 430\n";
    defines += "#define NUM_MULTISAMPLES " + std::to_string(mHDRBuffer->getMultiSamples()) + "\n";

    defines +=
        "#define TONE_MAPPING_MODE " + std::to_string(static_cast<int>(mToneMappingMode)) + "\n";

    std::string vert(utils::filesystem::loadToString("../share/resources/shaders/tonemap.vert"));
    std::string frag(utils::filesystem::loadToString("../share/resources/shaders/tonemap.frag"));

    mShader = std::make_unique<VistaGLSLShader>();
    mShader->InitVertexShaderFromString(defines + vert);
    mShader->InitFragmentShaderFromString(defines + frag);
    mShader->Link();

    mUniforms.exposure       = mShader->GetUniformLocation("uExposure");
    mUniforms.glareIntensity = mShader->GetUniformLocation("uGlareIntensity");

    mShaderDirty = false;
  }

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

  if (mGlareIntensity > 0) {
    mHDRBuffer->updateGlareMipMap();
  }

  if (doCalculateExposure && mEnableAutoExposure) {
    mExposure = glm::clamp(mAutoExposure, mMinAutoExposure, mMaxAutoExposure);
  }

  float exposure = std::pow(2.F, mExposure + mExposureCompensation);

  mHDRBuffer->unbind();
  mHDRBuffer->getCurrentWriteAttachment()->Bind(GL_TEXTURE0);
  mHDRBuffer->getDepthAttachment()->Bind(GL_TEXTURE1);
  mHDRBuffer->getGlareMipMap()->Bind(GL_TEXTURE2);

  mShader->Bind();
  mShader->SetUniform(mUniforms.exposure, exposure);
  mShader->SetUniform(mUniforms.glareIntensity, mGlareIntensity);

  glDrawArrays(GL_TRIANGLES, 0, 3);

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
