////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "math.cuh"
#include "with_atmosphere.cuh"
#include "tiff_utils.hpp"

#include <cstdint>
#include <iostream>

////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void drawPlanet(float* shadowMap, ShadowSettings settings, LimbDarkening limbDarkening) {
  uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
  uint32_t i = y * settings.size + x;

  if ((x >= settings.size) || (y >= settings.size)) {
    return;
  }

  auto angles = math::mapPixelToAngles(
      glm::ivec2(x, y), settings.size, settings.mappingExponent, settings.includeUmbra);

  glm::vec2 center   = 0.5f * glm::vec2(settings.size, settings.size);
  float     distance = glm::length(glm::vec2(x, y) - center);

  shadowMap[i*3+0] = distance < 0.5f * settings.size ? 1.0f : 0.0f;
  shadowMap[i*3+1] = 0.5;
  shadowMap[i*3+2] = 0.5;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void computeAtmosphereShadow(float* shadowMap, ShadowSettings settings, std::string const& atmosphereSettings,  LimbDarkening limbDarkening) {
// Compute the 2D kernel size.
  dim3     blockSize(16, 16);
  uint32_t numBlocksX = (settings.size + blockSize.x - 1) / blockSize.x;
  uint32_t numBlocksY = (settings.size + blockSize.y - 1) / blockSize.y;
  dim3     gridSize   = dim3(numBlocksX, numBlocksY);

  tiff_utils::RGBTexture multiscattering = tiff_utils::read3DTexture(atmosphereSettings + "/multiple_scattering.tif");
  tiff_utils::RGBTexture singleScattering = tiff_utils::read3DTexture(atmosphereSettings + "/single_aerosols_scattering.tif");
  tiff_utils::RGBTexture theta_deviation = tiff_utils::read2DTexture(atmosphereSettings + "/theta_deviation.tif");
  tiff_utils::RGBTexture phase = tiff_utils::read2DTexture(atmosphereSettings + "/phase.tif");

  std::cout << "Computing shadow map with atmosphere..." << std::endl;
  std::cout << "  - Mutli-scattering texture dimensions: " << multiscattering.width << "x" << multiscattering.height << "x" << multiscattering.depth << std::endl;
  std::cout << "  - Single-scattering texture dimensions: " << singleScattering.width << "x" << singleScattering.height << "x" << singleScattering.depth << std::endl;
  std::cout << "  - Theta deviation texture dimensions: " << theta_deviation.width << "x" << theta_deviation.height << std::endl;
  std::cout << "  - Phase texture dimensions: " << phase.width << "x" << phase.height << std::endl;

  drawPlanet<<<gridSize, blockSize>>>(shadowMap, settings, limbDarkening);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
