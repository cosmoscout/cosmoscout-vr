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

__device__ glm::vec2 intersectSphere(glm::dvec3 rayOrigin, glm::dvec3 rayDir, double radius) {
  double b   = glm::dot(rayOrigin, rayDir);
  double c   = glm::dot(rayOrigin, rayOrigin) - radius * radius;
  double det = b * b - c;

  if (det < 0.0) {
    return glm::dvec2(1, -1);
  }

  det = glm::sqrt(det);
  return glm::vec2(-b - det, -b + det);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

__device__ float safeSqrt(float a) {
  return std::sqrt(std::max(a, 0.0f));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

__device__ float clampDistance(float d) {
  return std::max(d, 0.0f);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

__device__ float getTextureCoordFromUnitRange(float x, int textureSize) {
  return 0.5 / float(textureSize) + x * (1.0 - 1.0 / float(textureSize));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

__device__ float distanceToTopAtmosphereBoundary(float r, float mu) {
  float discriminant = r * r * (mu * mu - 1.0) + TOP_RADIUS * TOP_RADIUS;
  return clampDistance(-r * mu + safeSqrt(discriminant));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

__device__ glm::vec2 getTransmittanceTextureUvFromRMu(float r, float mu) {
  // Distance to top atmosphere boundary for a horizontal ray at ground level.
  float H = sqrt(TOP_RADIUS * TOP_RADIUS - BOTTOM_RADIUS * BOTTOM_RADIUS);
  // Distance to the horizon.
  float rho = safeSqrt(r * r - BOTTOM_RADIUS * BOTTOM_RADIUS);
  // Distance to the top atmosphere boundary for the ray (r,mu), and its minimum
  // and maximum values over all mu - obtained for (r,1) and (r,mu_horizon).
  float d    = distanceToTopAtmosphereBoundary(r, mu);
  float dMin = TOP_RADIUS - r;
  float dMax = rho + H;
  float xMu  = (d - dMin) / (dMax - dMin);
  float xR   = rho / H;
  return glm::vec2(getTextureCoordFromUnitRange(xMu, TRANSMITTANCE_TEXTURE_WIDTH),
      getTextureCoordFromUnitRange(xR, TRANSMITTANCE_TEXTURE_HEIGHT));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

glm::vec4 getScatteringTextureUvwzFromRMuMuSNu(
    float r, float mu, float muS, float nu, bool rayRMuIntersectsGround) {

  // Distance to top atmosphere boundary for a horizontal ray at ground level.
  float H = sqrt(TOP_RADIUS * TOP_RADIUS - BOTTOM_RADIUS * BOTTOM_RADIUS);
  // Distance to the horizon.
  float rho = safeSqrt(r * r - BOTTOM_RADIUS * BOTTOM_RADIUS);
  float u_r = getTextureCoordFromUnitRange(rho / H, SCATTERING_TEXTURE_R_SIZE);

  // Discriminant of the quadratic equation for the intersections of the ray (r,mu) with the ground
  // (see rayIntersectsGround).
  float rMu          = r * mu;
  float discriminant = rMu * rMu - r * r + BOTTOM_RADIUS * BOTTOM_RADIUS;
  float uMu;
  if (rayRMuIntersectsGround) {
    // Distance to the ground for the ray (r,mu), and its minimum and maximum values over all mu -
    // obtained for (r,-1) and (r,mu_horizon).
    float d    = -rMu - safeSqrt(discriminant);
    float dMin = r - BOTTOM_RADIUS;
    float dMax = rho;
    uMu = 0.5 - 0.5 * getTextureCoordFromUnitRange(dMax == dMin ? 0.0 : (d - dMin) / (dMax - dMin),
                          SCATTERING_TEXTURE_MU_SIZE / 2);
  } else {
    // Distance to the top atmosphere boundary for the ray (r,mu), and its minimum and maximum
    // values over all mu - obtained for (r,1) and (r,mu_horizon).
    float d    = -rMu + safeSqrt(discriminant + H * H);
    float dMin = TOP_RADIUS - r;
    float dMax = rho + H;
    uMu        = 0.5 + 0.5 * getTextureCoordFromUnitRange(
                          (d - dMin) / (dMax - dMin), SCATTERING_TEXTURE_MU_SIZE / 2);
  }

  float d    = distanceToTopAtmosphereBoundary(BOTTOM_RADIUS, muS);
  float dMin = TOP_RADIUS - BOTTOM_RADIUS;
  float dMax = H;
  float a    = (d - dMin) / (dMax - dMin);
  float D    = distanceToTopAtmosphereBoundary(BOTTOM_RADIUS, MU_S_MIN);
  float A    = (D - dMin) / (dMax - dMin);
  // An ad-hoc function equal to 0 for muS = MU_S_MIN (because then d = D and thus a = A), equal to
  // 1 for muS = 1 (because then d = dMin and thus a = 0), and with a large slope around muS = 0,
  // to get more texture samples near the horizon.
  float uMuS = getTextureCoordFromUnitRange(
      std::max(1.0 - a / A, 0.0) / (1.0 + a), SCATTERING_TEXTURE_MU_S_SIZE);

  float uNu = (nu + 1.0) / 2.0;
  return glm::vec4(uNu, uMuS, uMu, u_r);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

__device__ glm::vec3 getTransmittanceToTopAtmosphereBoundary(
    cudaTextureObject_t transmittanceTexture, float r, float mu) {
  glm::vec2 uv = getTransmittanceTextureUvFromRMu(r, mu);
  return glm::vec3(tex2D(transmittanceTexture, uv));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

__device__ bool rayIntersectsGround(float r, float mu) {
  return mu < 0.0 && r * r * (mu * mu - 1.0) + BOTTOM_RADIUS * BOTTOM_RADIUS >= 0.0;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

glm::vec3 moleculePhaseFunction(float nu) {
  float theta = std::acos(nu) / M_PI; // 0<->1
  return tex2D(uPhaseTexture, glm::vec2(theta, 0.0)).rgb;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

glm::vec3 aerosolPhaseFunction(float nu) {
  float theta = std::acos(nu) / M_PI; // 0<->1
  return tex2D(uPhaseTexture, glm::vec2(theta, 1.0)).rgb;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

__device__ void getCombinedScattering(cudaTextureObject_t multipleScatteringTexture,
    cudaTextureObject_t singleAerosolsScatteringTexture, float r, float mu, float muS, float nu,
    bool rayRMuIntersectsGround, glm::vec3& multipleScattering,
    glm::vec3& singleAerosolsScattering) {
  glm::vec4 uvwz = getScatteringTextureUvwzFromRMuMuSNu(r, mu, muS, nu, rayRMuIntersectsGround);
  float     texCoordX = uvwz.x * float(SCATTERING_TEXTURE_NU_SIZE - 1);
  float     texX      = floor(texCoordX);
  float     lerp      = texCoordX - texX;
  glm::vec3 uvw0 = glm::vec3((texX + uvwz.y) / float(SCATTERING_TEXTURE_NU_SIZE), uvwz.z, uvwz.w);
  glm::vec3 uvw1 =
      glm::vec3((texX + 1.0 + uvwz.y) / float(SCATTERING_TEXTURE_NU_SIZE), uvwz.z, uvwz.w);

  multipleScattering       = glm::vec3(tex3D(multipleScatteringTexture, uvw0) * (1.0 - lerp) +
                                       tex3D(multipleScatteringTexture, uvw1) * lerp);
  singleAerosolsScattering = glm::vec3(tex3D(singleAerosolsScatteringTexture, uvw0) * (1.0 - lerp) +
                                       tex3D(singleAerosolsScatteringTexture, uvw1) * lerp);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

__device__ glm::vec3 getSkyRadiance(cudaTextureObject_t transmittanceTexture,
    cudaTextureObject_t                                 multipleScatteringTexture,
    cudaTextureObject_t singleAerosolsScatteringTexture, glm::vec3 camera, glm::vec3 viewRay,
    glm::vec3 sunDirection, glm::vec3& transmittance) {
  // Compute the distance to the top atmosphere boundary along the view ray, assuming the viewer is
  // in space (or NaN if the view ray does not intersect the atmosphere).
  float r                               = length(camera);
  float rmu                             = dot(camera, viewRay);
  float distanceToTopAtmosphereBoundary = -rmu - sqrt(rmu * rmu - r * r + TOP_RADIUS * TOP_RADIUS);
  // If the viewer is in space and the view ray intersects the atmosphere, move the viewer to the
  // top atmosphere boundary (along the view ray):
  if (distanceToTopAtmosphereBoundary > 0.0) {
    camera = camera + viewRay * distanceToTopAtmosphereBoundary;
    r      = TOP_RADIUS;
    rmu += distanceToTopAtmosphereBoundary;
  } else if (r > TOP_RADIUS) {
    // If the view ray does not intersect the atmosphere, simply return 0.
    transmittance = glm::vec3(1.0);
    return glm::vec3(0.0);
  }
  // Compute the r, mu, muS and nu parameters needed for the texture lookups.
  float mu                     = rmu / r;
  float muS                    = dot(camera, sunDirection) / r;
  float nu                     = dot(viewRay, sunDirection);
  bool  rayRMuIntersectsGround = rayIntersectsGround(r, mu);

  transmittance = rayRMuIntersectsGround
                      ? glm::vec3(0.0)
                      : getTransmittanceToTopAtmosphereBoundary(transmittanceTexture, r, mu);

  glm::vec3 multipleScattering;
  glm::vec3 singleAerosolsScattering;
  getCombinedScattering(multipleScatteringTexture, singleAerosolsScatteringTexture, r, mu, muS, nu,
      rayRMuIntersectsGround, multipleScattering, singleAerosolsScattering);

  return multipleScattering * moleculePhaseFunction(nu) +
         singleAerosolsScattering * aerosolPhaseFunction(nu);
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

  shadowMap[i * 3 + 0] = 0.0;
  shadowMap[i * 3 + 1] = 0.0;
  shadowMap[i * 3 + 2] = 0.0;

  double     radius       = 6371000.0;
  double     atmoHeight   = 100000.0;
  glm::dvec3 camera       = glm::dvec3(0.0, 0.0, 300000000.0);
  double     fieldOfView  = 0.02 * M_PI;
  glm::vec3  sunDirection = glm::vec3(1.0, 0.0, 0.0);

  // Compute the direction of the ray.
  double theta = (x / (double)settings.size - 0.5) * fieldOfView;
  double phi   = (y / (double)settings.size - 0.5) * fieldOfView;

  glm::dvec3 rayDir = glm::dvec3(sin(theta) * cos(phi), sin(phi), -cos(theta) * cos(phi));

  // Compute the intersection point with the atmosphere.
  glm::dvec2 atmoT = intersectSphere(camera, rayDir, radius + atmoHeight);
  if (atmoT.y > 0.0 && atmoT.x < atmoT.y) {
    shadowMap[i * 3 + 0] = 0.5;
    shadowMap[i * 3 + 1] = 0.5;
    shadowMap[i * 3 + 2] = 1.0;
  }

  glm::vec3 radiance = getSkyRadiance(uTransmittanceTexture, multiscatteringTexture,
      singleScatteringTexture, camera, rayDir, sunDirection, shadowMap[i]);
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
