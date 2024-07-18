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

struct Constants {
  double TOP_RADIUS;
  double BOTTOM_RADIUS;
  int    TRANSMITTANCE_TEXTURE_WIDTH;
  int    TRANSMITTANCE_TEXTURE_HEIGHT;
  int    SCATTERING_TEXTURE_MU_SIZE;
  int    SCATTERING_TEXTURE_MU_S_SIZE;
  int    SCATTERING_TEXTURE_NU_SIZE;
  double MU_S_MIN;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

cudaTextureObject_t createCudaTexture(tiff_utils::RGBATexture const& texture) {
  cudaArray* cuArray;
  auto       channelDesc = cudaCreateChannelDesc<float4>();

  cudaMallocArray(&cuArray, &channelDesc, texture.width, texture.height);
  cudaMemcpy2DToArray(cuArray, 0, 0, texture.data.data(), texture.width * sizeof(float) * 4,
      texture.width * sizeof(float) * 4, texture.height, cudaMemcpyHostToDevice);

  // Specify texture object parameters
  cudaResourceDesc resDesc = {};
  resDesc.resType          = cudaResourceTypeArray;
  resDesc.res.array.array  = cuArray;

  cudaTextureDesc texDesc  = {};
  texDesc.addressMode[0]   = cudaAddressModeClamp;
  texDesc.addressMode[1]   = cudaAddressModeClamp;
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

// Using acos is not very stable for small angles. This function is used to compute the angle
// between two vectors in a more stable way.
__device__ double angleBetweenVectors(glm::dvec3 u, glm::dvec3 v) {
  return 2.0 * glm::asin(0.5 * glm::length(u - v));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

__device__ glm::vec4 texture2D(cudaTextureObject_t tex, glm::vec2 uv) {
  auto data = tex2D<float4>(tex, uv.x, uv.y);
  return glm::vec4(data.x, data.y, data.z, data.w);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

__device__ float safeSqrt(float a) {
  return glm::sqrt(glm::max(a, 0.0f));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

__device__ float clampDistance(float d) {
  return glm::max(d, 0.0f);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

__device__ float getTextureCoordFromUnitRange(float x, int textureSize) {
  return 0.5 / float(textureSize) + x * (1.0 - 1.0 / float(textureSize));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

__device__ float distanceToTopAtmosphereBoundary(Constants const& constants, float r, float mu) {
  float discriminant = r * r * (mu * mu - 1.0) + constants.TOP_RADIUS * constants.TOP_RADIUS;
  return clampDistance(-r * mu + safeSqrt(discriminant));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// As we are always in outer space, this function does not need the r parameter.
__device__ glm::vec2 getTransmittanceTextureUvFromRMu(Constants const& constants, double mu) {
  // Distance to top atmosphere boundary for a horizontal ray at ground level.
  double H = sqrt(constants.TOP_RADIUS * constants.TOP_RADIUS -
                  constants.BOTTOM_RADIUS * constants.BOTTOM_RADIUS);
  // Distance to the top atmosphere boundary for the ray (r,mu), and its minimum
  // and maximum values over all mu - obtained for (r,1) and (r,mu_horizon).
  double d    = distanceToTopAtmosphereBoundary(constants, constants.TOP_RADIUS, mu);
  double dMax = 2.0 * H;
  double xMu  = d / dMax;
  return glm::vec2(getTextureCoordFromUnitRange(xMu, constants.TRANSMITTANCE_TEXTURE_WIDTH),
      getTextureCoordFromUnitRange(1.0, constants.TRANSMITTANCE_TEXTURE_HEIGHT));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// As we are always in outer space, this function does not need the r parameter.
__device__ glm::vec3 getScatteringTextureUvwFromRMuMuSNu(
    Constants const& constants, double mu, double muS, double nu, bool rayRMuIntersectsGround) {

  // Distance to top atmosphere boundary for a horizontal ray at ground level.
  double H = sqrt(constants.TOP_RADIUS * constants.TOP_RADIUS -
                  constants.BOTTOM_RADIUS * constants.BOTTOM_RADIUS);

  // Discriminant of the quadratic equation for the intersections of the ray (r,mu) with the ground
  // (see rayIntersectsGround).
  double rMu          = constants.TOP_RADIUS * mu;
  double discriminant = rMu * rMu - constants.TOP_RADIUS * constants.TOP_RADIUS +
                        constants.BOTTOM_RADIUS * constants.BOTTOM_RADIUS;
  double uMu;
  if (rayRMuIntersectsGround) {
    // Distance to the ground for the ray (r,mu), and its minimum and maximum values over all mu -
    // obtained for (r,-1) and (r,mu_horizon).
    double d    = -rMu - safeSqrt(discriminant);
    double dMin = constants.TOP_RADIUS - constants.BOTTOM_RADIUS;
    double dMax = H;
    uMu = 0.5 - 0.5 * getTextureCoordFromUnitRange(dMax == dMin ? 0.0 : (d - dMin) / (dMax - dMin),
                          constants.SCATTERING_TEXTURE_MU_SIZE / 2);
  } else {
    // Distance to the top atmosphere boundary for the ray (r,mu), and its minimum and maximum
    // values over all mu - obtained for (r,1) and (r,mu_horizon).
    double d    = -rMu + safeSqrt(discriminant + H * H);
    double dMax = 2.0 * H;
    uMu         = 0.5 +
          0.5 * getTextureCoordFromUnitRange(d / dMax, constants.SCATTERING_TEXTURE_MU_SIZE / 2);
  }

  double d    = distanceToTopAtmosphereBoundary(constants, constants.BOTTOM_RADIUS, muS);
  double dMin = constants.TOP_RADIUS - constants.BOTTOM_RADIUS;
  double dMax = H;
  double a    = (d - dMin) / (dMax - dMin);
  double D =
      distanceToTopAtmosphereBoundary(constants, constants.BOTTOM_RADIUS, constants.MU_S_MIN);
  double A = (D - dMin) / (dMax - dMin);
  // An ad-hoc function equal to 0 for muS = MU_S_MIN (because then d = D and thus a = A), equal to
  // 1 for muS = 1 (because then d = dMin and thus a = 0), and with a large slope around muS = 0,
  // to get more texture samples near the horizon.
  float uMuS = getTextureCoordFromUnitRange(
      glm::max(1.0 - a / A, 0.0) / (1.0 + a), constants.SCATTERING_TEXTURE_MU_S_SIZE);

  float uNu = (nu + 1.0) / 2.0;
  return glm::vec3(uNu, uMuS, uMu);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

__device__ glm::vec3 getTransmittanceToTopAtmosphereBoundary(
    Constants const& constants, cudaTextureObject_t transmittanceTexture, double mu) {
  glm::vec2 uv = getTransmittanceTextureUvFromRMu(constants, mu);
  return glm::vec3(texture2D(transmittanceTexture, uv));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

__device__ bool rayIntersectsGround(Constants const& constants, double mu) {
  return mu < 0.0 && constants.TOP_RADIUS * constants.TOP_RADIUS * (mu * mu - 1.0) +
                             constants.BOTTOM_RADIUS * constants.BOTTOM_RADIUS >=
                         0.0;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

__device__ glm::vec3 moleculePhaseFunction(cudaTextureObject_t phaseTexture, float nu) {
  float theta = glm::acos(nu) / M_PI; // 0<->1
  return glm::vec3(texture2D(phaseTexture, glm::vec2(theta, 0.0)));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

__device__ glm::vec3 aerosolPhaseFunction(cudaTextureObject_t phaseTexture, float nu) {
  float theta = glm::acos(nu) / M_PI; // 0<->1
  return glm::vec3(texture2D(phaseTexture, glm::vec2(theta, 1.0)));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

__device__ void getCombinedScattering(Constants const& constants,
    cudaTextureObject_t                                multipleScatteringTexture,
    cudaTextureObject_t singleAerosolsScatteringTexture, float mu, float muS, float nu,
    bool rayRMuIntersectsGround, glm::vec3& multipleScattering,
    glm::vec3& singleAerosolsScattering) {
  glm::vec3 uvw =
      getScatteringTextureUvwFromRMuMuSNu(constants, mu, muS, nu, rayRMuIntersectsGround);
  float     texCoordX = uvw.x * float(constants.SCATTERING_TEXTURE_NU_SIZE - 1);
  float     texX      = floor(texCoordX);
  float     lerp      = texCoordX - texX;
  glm::vec2 uv0 = glm::vec2((texX + uvw.y) / float(constants.SCATTERING_TEXTURE_NU_SIZE), uvw.z);
  glm::vec2 uv1 =
      glm::vec2((texX + 1.0 + uvw.y) / float(constants.SCATTERING_TEXTURE_NU_SIZE), uvw.z);

  multipleScattering = glm::vec3(texture2D(multipleScatteringTexture, uv0) * (1.0f - lerp) +
                                 texture2D(multipleScatteringTexture, uv1) * lerp);
  singleAerosolsScattering =
      glm::vec3(texture2D(singleAerosolsScatteringTexture, uv0) * (1.0f - lerp) +
                texture2D(singleAerosolsScatteringTexture, uv1) * lerp);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

__device__ glm::vec3 getSkyRadiance(Constants const& constants, cudaTextureObject_t phaseTexture,
    cudaTextureObject_t transmittanceTexture, cudaTextureObject_t multipleScatteringTexture,
    cudaTextureObject_t singleAerosolsScatteringTexture, glm::dvec3 camera, glm::dvec3 viewRay,
    glm::dvec3 sunDirection, glm::vec3& transmittance) {
  // Compute the distance to the top atmosphere boundary along the view ray, assuming the viewer is
  // in space (or NaN if the view ray does not intersect the atmosphere).
  double r   = length(camera);
  double rmu = dot(camera, viewRay);
  double distanceToTopAtmosphereBoundary =
      -rmu - sqrt(rmu * rmu - r * r + constants.TOP_RADIUS * constants.TOP_RADIUS);

  // If the view ray does not intersect the atmosphere, simply return 0.
  if (distanceToTopAtmosphereBoundary <= 0.0 && r > constants.TOP_RADIUS) {
    transmittance = glm::vec3(1.0);
    return glm::vec3(0.0);
  }

  camera += viewRay * distanceToTopAtmosphereBoundary;

  // Compute the mu, muS and nu parameters needed for the texture lookups.
  double mu                     = (rmu + distanceToTopAtmosphereBoundary) / constants.TOP_RADIUS;
  double muS                    = dot(camera, sunDirection) / constants.TOP_RADIUS;
  double nu                     = dot(viewRay, sunDirection);
  bool   rayRMuIntersectsGround = rayIntersectsGround(constants, mu);

  transmittance = rayRMuIntersectsGround ? glm::vec3(0.0)
                                         : getTransmittanceToTopAtmosphereBoundary(
                                               constants, transmittanceTexture, mu);

  glm::vec3 multipleScattering;
  glm::vec3 singleAerosolsScattering;
  getCombinedScattering(constants, multipleScatteringTexture, singleAerosolsScatteringTexture, mu,
      muS, nu, rayRMuIntersectsGround, multipleScattering, singleAerosolsScattering);

  return multipleScattering * moleculePhaseFunction(phaseTexture, nu) +
         singleAerosolsScattering * aerosolPhaseFunction(phaseTexture, nu);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void drawPlanet(float* shadowMap, ShadowSettings settings, LimbDarkening limbDarkening,
    Constants constants, cudaTextureObject_t multiscatteringTexture,
    cudaTextureObject_t singleScatteringTexture, cudaTextureObject_t thetaDeviationTexture,
    cudaTextureObject_t phaseTexture, cudaTextureObject_t transmittanceTexture) {

  uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
  uint32_t i = y * settings.size + x;

  if ((x >= settings.size) || (y >= settings.size)) {
    return;
  }

  // Horizon close up from quite close to the Earth.
  // glm::dvec3 camera       = glm::dvec3(0.0, constants.BOTTOM_RADIUS, 1000000.0);
  // double     fieldOfView  = 0.05 * M_PI;
  // glm::dvec3 sunDirection = glm::normalize(glm::vec3(0.0, 0.0, -1.0));
  // float      exposure     = 0.001;

  // Horizon close up from Moon.
  glm::dvec3 camera       = glm::dvec3(0.0, constants.BOTTOM_RADIUS, 300000000.0);
  double     fieldOfView  = 0.005 * M_PI;
  glm::dvec3 sunDirection = glm::normalize(glm::vec3(0.0, 0.0, -1.0));
  float      exposure     = 0.0001;

  // Total eclipse from Moon.
  // glm::dvec3 camera       = glm::dvec3(0.0, 0.0, 300000000.0);
  // double     fieldOfView  = 0.018 * M_PI;
  // glm::dvec3 sunDirection = glm::normalize(glm::vec3(0.0, 0.0, -1.0));
  // float      exposure     = 0.001;

  // Compute the direction of the ray.
  double theta = (x / (double)settings.size - 0.5) * fieldOfView;
  double phi   = (y / (double)settings.size - 0.5) * fieldOfView;

  glm::dvec3 rayDir =
      glm::dvec3(glm::sin(theta) * glm::cos(phi), glm::sin(phi), -glm::cos(theta) * glm::cos(phi));
  glm::vec3 transmittance;

  glm::vec3 skyRadiance = getSkyRadiance(constants, phaseTexture, transmittanceTexture,
      multiscatteringTexture, singleScatteringTexture, camera, rayDir, sunDirection, transmittance);

  float sunAngularRadius = 0.0082 / 2.0;

  glm::vec3 sunRadiance =
      transmittance *
      glm::vec3(
          limbDarkening.get(angleBetweenVectors(rayDir, sunDirection) / sunAngularRadius) * 1.1e9);

  shadowMap[i * 3 + 0] = (skyRadiance.r + sunRadiance.r) * exposure;
  shadowMap[i * 3 + 1] = (skyRadiance.g + sunRadiance.g) * exposure;
  shadowMap[i * 3 + 2] = (skyRadiance.b + sunRadiance.b) * exposure;
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
      tiff_utils::read2DTexture(atmosphereSettings + "/multiple_scattering.tif", 31);
  tiff_utils::RGBATexture singleScattering =
      tiff_utils::read2DTexture(atmosphereSettings + "/single_aerosols_scattering.tif", 31);
  tiff_utils::RGBATexture theta_deviation =
      tiff_utils::read2DTexture(atmosphereSettings + "/theta_deviation.tif");
  tiff_utils::RGBATexture phase = tiff_utils::read2DTexture(atmosphereSettings + "/phase.tif");
  tiff_utils::RGBATexture transmittance =
      tiff_utils::read2DTexture(atmosphereSettings + "/transmittance.tif");

  std::cout << "Computing shadow map with atmosphere..." << std::endl;
  std::cout << "  - Mutli-scattering texture dimensions: " << multiscattering.width << "x"
            << multiscattering.height << std::endl;
  std::cout << "  - Single-scattering texture dimensions: " << singleScattering.width << "x"
            << singleScattering.height << std::endl;
  std::cout << "  - Theta deviation texture dimensions: " << theta_deviation.width << "x"
            << theta_deviation.height << std::endl;
  std::cout << "  - Phase texture dimensions: " << phase.width << "x" << phase.height << std::endl;
  std::cout << "  - Transmittance texture dimensions: " << transmittance.width << "x"
            << transmittance.height << std::endl;

  cudaTextureObject_t multiscatteringTexture  = createCudaTexture(multiscattering);
  cudaTextureObject_t singleScatteringTexture = createCudaTexture(singleScattering);
  cudaTextureObject_t thetaDeviationTexture   = createCudaTexture(theta_deviation);
  cudaTextureObject_t phaseTexture            = createCudaTexture(phase);
  cudaTextureObject_t transmittanceTexture    = createCudaTexture(transmittance);

  Constants constants;
  constants.BOTTOM_RADIUS                = 6371000.0;
  constants.TOP_RADIUS                   = 6371000.0 + 100000.0;
  constants.TRANSMITTANCE_TEXTURE_WIDTH  = 256;
  constants.TRANSMITTANCE_TEXTURE_HEIGHT = 64;
  constants.SCATTERING_TEXTURE_MU_SIZE   = 128;
  constants.SCATTERING_TEXTURE_MU_S_SIZE = 256 / 8;
  constants.SCATTERING_TEXTURE_NU_SIZE   = 8;
  constants.MU_S_MIN                     = std::cos(2.094395160675049);

  drawPlanet<<<gridSize, blockSize>>>(shadowMap, settings, limbDarkening, constants,
      multiscatteringTexture, singleScatteringTexture, thetaDeviationTexture, phaseTexture,
      transmittanceTexture);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
