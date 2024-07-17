////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "gpuErrCheck.hpp"
#include "math.cuh"
#include "tiff_utils.hpp"
#include "with_atmosphere.cuh"

#include <cstdint>
#include <iostream>

////////////////////////////////////////////////////////////////////////////////////////////////////

cudaTextureObject_t createCudaTexture(tiff_utils::RGBATexture const& texture) {
  cudaArray* cuArray;
  auto       channelDesc = cudaCreateChannelDesc<float4>();

  if (texture.depth == 1) {
    cudaMallocArray(&cuArray, &channelDesc, texture.width, texture.height);
    cudaMemcpy2DToArray(cuArray, 0, 0, texture.data.data(), texture.width * sizeof(float) * 4,
        texture.width * sizeof(float) * 4, texture.height, cudaMemcpyHostToDevice);
  } else {
    cudaMalloc3DArray(
        &cuArray, &channelDesc, make_cudaExtent(texture.width, texture.height, texture.depth));
    cudaMemcpy3DParms copyParams = {};
    copyParams.srcPtr            = make_cudaPitchedPtr((void*)texture.data.data(),
                   texture.width * sizeof(float) * 4, texture.width, texture.height);
    copyParams.dstArray          = cuArray;
    copyParams.extent            = make_cudaExtent(texture.width, texture.height, texture.depth);
    copyParams.kind              = cudaMemcpyHostToDevice;
    cudaMemcpy3D(&copyParams);
  }

  // Specify texture object parameters
  cudaResourceDesc resDesc = {};
  resDesc.resType          = cudaResourceTypeArray;
  resDesc.res.array.array  = cuArray;

  cudaTextureDesc texDesc  = {};
  texDesc.addressMode[0]   = cudaAddressModeClamp;
  texDesc.addressMode[1]   = cudaAddressModeClamp;
  texDesc.addressMode[3]   = cudaAddressModeClamp;
  texDesc.filterMode       = cudaFilterModeLinear;
  texDesc.readMode         = cudaReadModeElementType;
  texDesc.normalizedCoords = 1;

  // Create texture object
  cudaTextureObject_t textureObject = 0;
  cudaCreateTextureObject(&textureObject, &resDesc, &texDesc, nullptr);

  gpuErrchk(cudaGetLastError());

  return textureObject;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void drawPlanet(float* shadowMap, ShadowSettings settings, LimbDarkening limbDarkening,
    cudaTextureObject_t multiscatteringTexture, cudaTextureObject_t singleScatteringTexture,
    cudaTextureObject_t thetaDeviationTexture, cudaTextureObject_t phaseTexture) {

  uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
  uint32_t i = y * settings.size + x;

  if ((x >= settings.size) || (y >= settings.size)) {
    return;
  }

  // Sample the thetaDeviationTexture.
  float2 texCoords      = make_float2(x / (float)settings.size, y / (float)settings.size);
  float4 thetaDeviation = tex2D<float4>(thetaDeviationTexture, texCoords.x, texCoords.y);

  shadowMap[i * 3 + 0] = thetaDeviation.x * 100.f;
  shadowMap[i * 3 + 1] = thetaDeviation.y * 100.f;
  shadowMap[i * 3 + 2] = thetaDeviation.z * 100.f;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void computeAtmosphereShadow(float* shadowMap, ShadowSettings settings,
    std::string const& atmosphereSettings, LimbDarkening limbDarkening) {
  // Compute the 2D kernel size.
  dim3     blockSize(16, 16);
  uint32_t numBlocksX = (settings.size + blockSize.x - 1) / blockSize.x;
  uint32_t numBlocksY = (settings.size + blockSize.y - 1) / blockSize.y;
  dim3     gridSize   = dim3(numBlocksX, numBlocksY);

  tiff_utils::RGBATexture multiscattering =
      tiff_utils::read3DTexture(atmosphereSettings + "/multiple_scattering.tif");
  tiff_utils::RGBATexture singleScattering =
      tiff_utils::read3DTexture(atmosphereSettings + "/single_aerosols_scattering.tif");
  tiff_utils::RGBATexture theta_deviation =
      tiff_utils::read2DTexture(atmosphereSettings + "/theta_deviation.tif");
  tiff_utils::RGBATexture phase = tiff_utils::read2DTexture(atmosphereSettings + "/phase.tif");

  std::cout << "Computing shadow map with atmosphere..." << std::endl;
  std::cout << "  - Mutli-scattering texture dimensions: " << multiscattering.width << "x"
            << multiscattering.height << "x" << multiscattering.depth << std::endl;
  std::cout << "  - Single-scattering texture dimensions: " << singleScattering.width << "x"
            << singleScattering.height << "x" << singleScattering.depth << std::endl;
  std::cout << "  - Theta deviation texture dimensions: " << theta_deviation.width << "x"
            << theta_deviation.height << std::endl;
  std::cout << "  - Phase texture dimensions: " << phase.width << "x" << phase.height << std::endl;

  cudaTextureObject_t multiscatteringTexture  = createCudaTexture(multiscattering);
  cudaTextureObject_t singleScatteringTexture = createCudaTexture(singleScattering);
  cudaTextureObject_t thetaDeviationTexture   = createCudaTexture(theta_deviation);
  cudaTextureObject_t phaseTexture            = createCudaTexture(phase);

  drawPlanet<<<gridSize, blockSize>>>(shadowMap, settings, limbDarkening, multiscatteringTexture,
      singleScatteringTexture, thetaDeviationTexture, phaseTexture);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
