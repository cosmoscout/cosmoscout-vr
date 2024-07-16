////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "without_atmosphere.cuh"
#include "math.cuh"

#include <cstdint>

////////////////////////////////////////////////////////////////////////////////////////////////////

__device__ void writeAsRGBValue (float value, float* shadowMap, uint32_t index) {
  shadowMap[index * 3 + 0] = value;
  shadowMap[index * 3 + 1] = value;
  shadowMap[index * 3 + 2] = value;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void computeLimbDarkeningShadow(float* shadowMap, ShadowSettings settings, LimbDarkening limbDarkening) {
  uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
  uint32_t i = y * settings.size + x;

  if ((x >= settings.size) || (y >= settings.size)) {
    return;
  }

  auto angles = math::mapPixelToAngles(glm::ivec2(x, y), settings.size,
      settings.mappingExponent, settings.includeUmbra);

  double sunArea = math::getCircleArea(1.0);

  float intensity = static_cast<float>(
      1 - math::sampleCircleIntersection(1.0, angles.x, angles.y, limbDarkening) / sunArea);

  writeAsRGBValue(intensity, shadowMap, i);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void computeCircleIntersectionShadow(float* shadowMap, ShadowSettings settings) {
  uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
  uint32_t i = y * settings.size + x;

  if ((x >= settings.size) || (y >= settings.size)) {
    return;
  }

  auto angles = math::mapPixelToAngles(glm::ivec2(x, y), settings.size,
      settings.mappingExponent, settings.includeUmbra);

  double sunArea = math::getCircleArea(1.0);

  float intensity = static_cast<float>(1.0 - math::getCircleIntersection(1.0, angles.x, angles.y) / sunArea);

  writeAsRGBValue(intensity, shadowMap, i);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void computeLinearShadow(float* shadowMap, ShadowSettings settings) {
  uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
  uint32_t i = y * settings.size + x;

  if ((x >= settings.size) || (y >= settings.size)) {
    return;
  }

  auto angles = math::mapPixelToAngles(glm::ivec2(x, y), settings.size,
      settings.mappingExponent, settings.includeUmbra);

  double phiSun = 1.0;
  double phiOcc = angles[0];
  double delta  = angles[1];

  double visiblePortion =
      (delta - glm::abs(phiSun - phiOcc)) / (phiSun + phiOcc - glm::abs(phiSun - phiOcc));

  double maxDepth = glm::min(1.0, glm::pow(phiOcc / phiSun, 2.0));

  float intensity = static_cast<float>(1.0 - maxDepth * glm::clamp(1.0 - visiblePortion, 0.0, 1.0));

  writeAsRGBValue(intensity, shadowMap, i);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void computeSmoothstepShadow(float* shadowMap, ShadowSettings settings) {
  uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
  uint32_t i = y * settings.size + x;

  if ((x >= settings.size) || (y >= settings.size)) {
    return;
  }

  auto angles = math::mapPixelToAngles(glm::ivec2(x, y), settings.size,
      settings.mappingExponent, settings.includeUmbra);

  double phiSun = 1.0;
  double phiOcc = angles[0];
  double delta  = angles[1];

  double visiblePortion =
      (delta - glm::abs(phiSun - phiOcc)) / (phiSun + phiOcc - glm::abs(phiSun - phiOcc));

  double maxDepth = glm::min(1.0, glm::pow(phiOcc / phiSun, 2.0));

  float intensity = static_cast<float>(
      1.0 - maxDepth * glm::clamp(1.0 - glm::smoothstep(0.0, 1.0, visiblePortion), 0.0, 1.0));

  writeAsRGBValue(intensity, shadowMap, i);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
