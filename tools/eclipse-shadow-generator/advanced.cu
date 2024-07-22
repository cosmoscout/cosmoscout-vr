////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "advanced.cuh"
#include "common.hpp"
#include "gpuErrCheck.hpp"
#include "math.cuh"
#include "tiff_utils.hpp"

#include <stb_image_write.h>

#include <cstdint>
#include <iostream>

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace {

////////////////////////////////////////////////////////////////////////////////////////////////////

enum class Mode { eBruneton, ePlanetView, eAtmoView };

////////////////////////////////////////////////////////////////////////////////////////////////////

struct GeometrySettings {
  double phiSun     = 0.0082 / 2.0;
  double phiOcc     = 0.02;
  double radiusOcc  = 6370900.0;
  double radiusAtmo = 6451000.0;
  double delta      = 0.02;
};

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

// Tonemapping code and color space conversions.
// http://filmicworlds.com/blog/filmic-tonemapping-operators/

__device__ glm::vec3 uncharted2Tonemap(glm::vec3 c) {
  const float A = 0.15;
  const float B = 0.50;
  const float C = 0.10;
  const float D = 0.20;
  const float E = 0.02;
  const float F = 0.30;
  return ((c * (A * c + C * B) + D * E) / (c * (A * c + B) + D * F)) - E / F;
}

__device__ glm::vec3 tonemap(glm::vec3 c) {
  const float W        = 11.2;
  c                    = uncharted2Tonemap(10.0f * c);
  glm::vec3 whiteScale = glm::vec3(1.0) / uncharted2Tonemap(glm::vec3(W));
  return c * whiteScale;
}

__device__ float linearToSRGB(float c) {
  if (c <= 0.0031308f)
    return 12.92f * c;
  else
    return 1.055f * pow(c, 1.0f / 2.4f) - 0.055f;
}

__device__ glm::vec3 linearToSRGB(glm::vec3 c) {
  return glm::vec3(linearToSRGB(c.r), linearToSRGB(c.g), linearToSRGB(c.b));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

__device__ glm::vec4 texture2D(cudaTextureObject_t tex, glm::vec2 uv) {
  auto data = tex2D<float4>(tex, uv.x, uv.y);
  return glm::vec4(data.x, data.y, data.z, data.w);
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

// Rodrigues' rotation formula
__device__ glm::dvec3 rotateVector(glm::dvec3 v, glm::dvec3 a, double cosMu) {
  double sinMu = glm::sqrt(1.0 - cosMu * cosMu);
  return v * cosMu + glm::cross(a, v) * sinMu + a * glm::dot(a, v) * (1.0 - cosMu);
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

__device__ void getRefractedViewRays(Constants const& constants,
    cudaTextureObject_t thetaDeviationTexture, glm::dvec3 camera, glm::dvec3 viewRay,
    glm::dvec3& viewRayR, glm::dvec3& viewRayG, glm::dvec3& viewRayB) {

  double    mu = dot(camera, viewRay) / constants.TOP_RADIUS;
  glm::vec2 uv = getTransmittanceTextureUvFromRMu(constants, mu);

  // Cosine of the angular deviation of the ray due to refraction.
  glm::dvec3 muRGB = glm::cos(glm::dvec3(texture2D(thetaDeviationTexture, uv)));
  glm::dvec3 axis  = glm::normalize(glm::cross(camera, viewRay));

  viewRayR = normalize(rotateVector(viewRay, axis, muRGB.r));
  viewRayG = normalize(rotateVector(viewRay, axis, muRGB.g));
  viewRayB = normalize(rotateVector(viewRay, axis, muRGB.b));
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

__device__ glm::vec3 getLuminance(Constants const& constants, LimbDarkening limbDarkening,
    cudaTextureObject_t phaseTexture, cudaTextureObject_t thetaDeviationTexture,
    cudaTextureObject_t transmittanceTexture, cudaTextureObject_t multipleScatteringTexture,
    cudaTextureObject_t singleAerosolsScatteringTexture, glm::dvec3 camera, glm::dvec3 viewRay,
    glm::dvec3 sunDirection, double phiSun) {
  // Compute the distance to the top atmosphere boundary along the view ray, assuming the viewer is
  // in space (or NaN if the view ray does not intersect the atmosphere).
  double r   = length(camera);
  double rmu = dot(camera, viewRay);
  double distanceToTopAtmosphereBoundary =
      -rmu - sqrt(rmu * rmu - r * r + constants.TOP_RADIUS * constants.TOP_RADIUS);

  glm::vec3  skyLuminance  = glm::vec3(0.0);
  glm::vec3  transmittance = glm::vec3(1.0);
  glm::dvec3 viewRayR, viewRayG, viewRayB;
  viewRayR = viewRayG = viewRayB = viewRay;

  // We only need to compute the luminance if the view ray intersects the atmosphere.
  if (distanceToTopAtmosphereBoundary > 0.0) {

    camera += viewRay * distanceToTopAtmosphereBoundary;

    // Compute the mu, muS and nu parameters needed for the texture lookups.
    double mu                     = (rmu + distanceToTopAtmosphereBoundary) / constants.TOP_RADIUS;
    double muS                    = dot(camera, sunDirection) / constants.TOP_RADIUS;
    double nu                     = dot(viewRay, sunDirection);
    bool   rayRMuIntersectsGround = rayIntersectsGround(constants, mu);

    glm::vec3 multipleScattering;
    glm::vec3 singleAerosolsScattering;
    getCombinedScattering(constants, multipleScatteringTexture, singleAerosolsScatteringTexture, mu,
        muS, nu, rayRMuIntersectsGround, multipleScattering, singleAerosolsScattering);

    skyLuminance = multipleScattering * moleculePhaseFunction(phaseTexture, nu) +
                   singleAerosolsScattering * aerosolPhaseFunction(phaseTexture, nu);

    getRefractedViewRays(
        constants, thetaDeviationTexture, camera, viewRay, viewRayR, viewRayG, viewRayB);

    transmittance = rayRMuIntersectsGround ? glm::vec3(0.0)
                                           : getTransmittanceToTopAtmosphereBoundary(
                                                 constants, transmittanceTexture, mu);
  }

  float sunR = limbDarkening.get(angleBetweenVectors(viewRayR, sunDirection) / phiSun);
  float sunG = limbDarkening.get(angleBetweenVectors(viewRayG, sunDirection) / phiSun);
  float sunB = limbDarkening.get(angleBetweenVectors(viewRayB, sunDirection) / phiSun);

  glm::vec3 sunLuminance = transmittance * 1.1e9f * glm::vec3(sunR, sunG, sunB);

  return skyLuminance + sunLuminance;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void computeShadowMap(float* shadowMap, common::ShadowSettings settings,
    GeometrySettings geometrySettings, float exposure, common::OutputSettings output,
    LimbDarkening limbDarkening, Constants constants, cudaTextureObject_t multiscatteringTexture,
    cudaTextureObject_t singleScatteringTexture, cudaTextureObject_t thetaDeviationTexture,
    cudaTextureObject_t phaseTexture, cudaTextureObject_t transmittanceTexture) {

  uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
  uint32_t i = y * output.size + x;

  if ((x >= output.size) || (y >= output.size)) {
    return;
  }

  uint32_t  samplesX = 64;
  uint32_t  samplesY = 64;
  glm::vec3 illuminance(0.0);

  glm::dvec2 angles = math::mapPixelToAngles(
      glm::ivec2(x, y), output.size, settings.mappingExponent, settings.includeUmbra);

  double phiOcc = angles.x;
  double delta  = angles.y;

  double occDist    = geometrySettings.radiusOcc / glm::sin(phiOcc);
  double atmoRadius = geometrySettings.radiusAtmo;
  double phiAtmo    = glm::asin(atmoRadius / occDist);

  glm::dvec3 camera       = glm::dvec3(0.0, 0.0, occDist);
  glm::dvec3 sunDirection = glm::dvec3(0.0, glm::sin(delta), -glm::cos(delta));

  // Compute the direction of the ray.

  for (uint32_t sampleY = 0; sampleY < samplesY; ++sampleY) {
    double upperAltitude    = ((double)sampleY + 1.0) / samplesY;
    double lowerAltitude    = ((double)sampleY) / samplesY;
    double upperPhiRay      = phiOcc + upperAltitude * (phiAtmo - phiOcc);
    double lowerPhiRay      = phiOcc + lowerAltitude * (phiAtmo - phiOcc);
    double solidAnglePerRow = (math::getCapArea(upperPhiRay) - math::getCapArea(lowerPhiRay));

    for (uint32_t sampleX = 0; sampleX < samplesX; ++sampleX) {

      double theta    = (((double)sampleX + 0.5) / samplesX) * M_PI;
      double altitude = (((double)sampleY + 0.5) / samplesY);

      double     phiRay = phiOcc + altitude * (phiAtmo - phiOcc);
      glm::dvec3 rayDir = glm::dvec3(0.0, glm::sin(phiRay), -glm::cos(phiRay));
      rayDir = glm::normalize(rotateVector(rayDir, glm::dvec3(0.0, 0.0, -1.0), glm::cos(theta)));

      glm::vec3 luminance = getLuminance(constants, limbDarkening, phaseTexture,
          thetaDeviationTexture, transmittanceTexture, multiscatteringTexture,
          singleScatteringTexture, camera, rayDir, sunDirection, 1.0);

      illuminance += luminance * (float)solidAnglePerRow;
    }
  }

  illuminance = illuminance / (float)samplesX;

  illuminance = linearToSRGB(tonemap(illuminance * exposure));

  shadowMap[i * 3 + 0] = illuminance.r;
  shadowMap[i * 3 + 1] = illuminance.g;
  shadowMap[i * 3 + 2] = illuminance.b;
}
////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void drawAtmoView(float* shadowMap, common::ShadowSettings settings,
    GeometrySettings geometrySettings, float exposure, common::OutputSettings output,
    LimbDarkening limbDarkening, Constants constants, cudaTextureObject_t multiscatteringTexture,
    cudaTextureObject_t singleScatteringTexture, cudaTextureObject_t thetaDeviationTexture,
    cudaTextureObject_t phaseTexture, cudaTextureObject_t transmittanceTexture) {

  uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
  uint32_t i = y * output.size + x;

  if ((x >= output.size) || (y >= output.size)) {
    return;
  }

  double occDist    = geometrySettings.radiusOcc / glm::sin(geometrySettings.phiOcc);
  double atmoRadius = geometrySettings.radiusAtmo;
  double phiAtmo    = glm::asin(atmoRadius / occDist);

  glm::dvec3 camera = glm::dvec3(0.0, 0.0, occDist);
  glm::dvec3 sunDirection =
      glm::dvec3(0.0, glm::sin(geometrySettings.delta), -glm::cos(geometrySettings.delta));

  // Compute the direction of the ray.
  double theta    = (x / (double)output.size) * M_PI;
  double altitude = (y / (double)output.size);

  double     phiRay = geometrySettings.phiOcc + altitude * (phiAtmo - geometrySettings.phiOcc);
  glm::dvec3 rayDir = glm::dvec3(0.0, glm::sin(phiRay), -glm::cos(phiRay));
  rayDir = glm::normalize(rotateVector(rayDir, glm::dvec3(0.0, 0.0, -1.0), glm::cos(theta)));

  glm::vec3 luminance = getLuminance(constants, limbDarkening, phaseTexture, thetaDeviationTexture,
      transmittanceTexture, multiscatteringTexture, singleScatteringTexture, camera, rayDir,
      sunDirection, geometrySettings.phiSun);

  luminance = linearToSRGB(tonemap(luminance * exposure));

  shadowMap[i * 3 + 0] = luminance.r;
  shadowMap[i * 3 + 1] = luminance.g;
  shadowMap[i * 3 + 2] = luminance.b;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void drawPlanet(float* shadowMap, common::ShadowSettings settings,
    GeometrySettings geometrySettings, float exposure, common::OutputSettings output,
    LimbDarkening limbDarkening, Constants constants, cudaTextureObject_t multiscatteringTexture,
    cudaTextureObject_t singleScatteringTexture, cudaTextureObject_t thetaDeviationTexture,
    cudaTextureObject_t phaseTexture, cudaTextureObject_t transmittanceTexture) {

  uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
  uint32_t i = y * output.size + x;

  if ((x >= output.size) || (y >= output.size)) {
    return;
  }

  // Total eclipse from Moon, horizon close up.
  double     occDist     = geometrySettings.radiusOcc / glm::sin(geometrySettings.phiOcc);
  glm::dvec3 camera      = glm::dvec3(0.0, 0.0, occDist);
  double     fieldOfView = 0.02 * M_PI;
  glm::dvec3 sunDirection =
      glm::dvec3(0.0, glm::sin(geometrySettings.delta), -glm::cos(geometrySettings.delta));

  // Compute the direction of the ray.
  double theta = (x / (double)output.size - 0.5) * fieldOfView;
  double phi   = (y / (double)output.size - 0.5) * fieldOfView;

  glm::dvec3 rayDir =
      glm::dvec3(glm::sin(theta) * glm::cos(phi), glm::sin(phi), -glm::cos(theta) * glm::cos(phi));

  glm::vec3 luminance = getLuminance(constants, limbDarkening, phaseTexture, thetaDeviationTexture,
      transmittanceTexture, multiscatteringTexture, singleScatteringTexture, camera, rayDir,
      sunDirection, geometrySettings.phiSun);

  luminance = linearToSRGB(tonemap(luminance * exposure));

  shadowMap[i * 3 + 0] = luminance.r;
  shadowMap[i * 3 + 1] = luminance.g;
  shadowMap[i * 3 + 2] = luminance.b;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void addAtmosphereSettingsFlags(cs::utils::CommandLine& commandLine, std::string& settings) {
  commandLine.addArgument(
      {"--atmosphere-settings"}, &settings, "The path to the atmosphere settings directory.");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void addGeometrySettingsFlags(cs::utils::CommandLine& commandLine, GeometrySettings& settings) {
  commandLine.addArgument({"--phi-sun"}, &settings.phiSun,
      "The angular radius of the sun. Default is " + std::to_string(settings.phiSun));
  commandLine.addArgument({"--phi-occ"}, &settings.phiOcc,
      "The angular radius of the occluding body. Default is " + std::to_string(settings.phiOcc));
  commandLine.addArgument({"--delta"}, &settings.delta,
      "The angular distance between the centers of the sun and the occluding body. Default is " +
          std::to_string(settings.delta));
  commandLine.addArgument({"--radius-occ"}, &settings.radiusOcc,
      "The radius of the occluding body. Default is " + std::to_string(settings.radiusOcc));
  commandLine.addArgument({"--radius-atmo"}, &settings.radiusAtmo,
      "The radius of the atmosphere. Default is " + std::to_string(settings.radiusAtmo));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

int run(Mode mode, std::vector<std::string> const& arguments) {

  common::ShadowSettings shadow;
  common::OutputSettings output;
  std::string            atmosphereSettings;
  float                  exposure = 0.0001;
  GeometrySettings       geometrySettings;
  bool                   cPrintHelp = false;

  // First configure all possible command line options.
  cs::utils::CommandLine args("Here are the available options:");
  common::addShadowSettingsFlags(args, shadow);
  common::addOutputSettingsFlags(args, output);
  addAtmosphereSettingsFlags(args, atmosphereSettings);
  addGeometrySettingsFlags(args, geometrySettings);
  args.addArgument({"--exposure"}, &exposure,
      "The exposure of the image. Default is " + std::to_string(exposure));
  args.addArgument({"-h", "--help"}, &cPrintHelp, "Show this help message.");

  // Then do the actual parsing.
  try {
    args.parse(arguments);
  } catch (std::runtime_error const& e) {
    std::cerr << "Failed to parse command line arguments: " << e.what() << std::endl;
    return 1;
  }

  // When cPrintHelp was set to true, we print a help message and exit.
  if (cPrintHelp) {
    args.printHelp();
    return 0;
  }

  // If we are in atmosphere mode, we need also the atmosphere settings.
  if (atmosphereSettings.empty()) {
    std::cerr << "When using the 'with-atmosphere' mode, you must provide the path to the "
                 "atmosphere settings directory using --atmosphere-settings!"
              << std::endl;
    return 1;
  }

  // Initialize the limb darkening model.
  LimbDarkening limbDarkening;
  limbDarkening.init();

  // Compute the 2D kernel size.
  dim3     blockSize(16, 16);
  uint32_t numBlocksX = (output.size + blockSize.x - 1) / blockSize.x;
  uint32_t numBlocksY = (output.size + blockSize.y - 1) / blockSize.y;
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
  constants.BOTTOM_RADIUS                = geometrySettings.radiusOcc;
  constants.TOP_RADIUS                   = geometrySettings.radiusAtmo;
  constants.TRANSMITTANCE_TEXTURE_WIDTH  = 256;
  constants.TRANSMITTANCE_TEXTURE_HEIGHT = 64;
  constants.SCATTERING_TEXTURE_MU_SIZE   = 128;
  constants.SCATTERING_TEXTURE_MU_S_SIZE = 256 / 8;
  constants.SCATTERING_TEXTURE_NU_SIZE   = 8;
  constants.MU_S_MIN                     = std::cos(2.094395160675049);

  // Allocate the shared memory for the shadow map.
  float* shadowMap = nullptr;
  gpuErrchk(cudaMallocManaged(
      &shadowMap, static_cast<size_t>(output.size * output.size) * 3 * sizeof(float)));

  if (mode == Mode::eBruneton) {
    computeShadowMap<<<gridSize, blockSize>>>(shadowMap, shadow, geometrySettings, exposure, output,
        limbDarkening, constants, multiscatteringTexture, singleScatteringTexture,
        thetaDeviationTexture, phaseTexture, transmittanceTexture);
  } else if (mode == Mode::ePlanetView) {
    drawPlanet<<<gridSize, blockSize>>>(shadowMap, shadow, geometrySettings, exposure, output,
        limbDarkening, constants, multiscatteringTexture, singleScatteringTexture,
        thetaDeviationTexture, phaseTexture, transmittanceTexture);
  } else if (mode == Mode::eAtmoView) {
    drawAtmoView<<<gridSize, blockSize>>>(shadowMap, shadow, geometrySettings, exposure, output,
        limbDarkening, constants, multiscatteringTexture, singleScatteringTexture,
        thetaDeviationTexture, phaseTexture, transmittanceTexture);
  }

  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());

  // Finally write the output texture!
  stbi_write_hdr(output.output.c_str(), static_cast<int>(output.size),
      static_cast<int>(output.size), 3, shadowMap);

  // Free the shared memory.
  gpuErrchk(cudaFree(shadowMap));

  return 0;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace advanced {

////////////////////////////////////////////////////////////////////////////////////////////////////

int brunetonMode(std::vector<std::string> const& arguments) {
  return run(Mode::eBruneton, arguments);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

int planetViewMode(std::vector<std::string> const& arguments) {
  return run(Mode::ePlanetView, arguments);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

int atmoViewMode(std::vector<std::string> const& arguments) {
  return run(Mode::eAtmoView, arguments);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace advanced

////////////////////////////////////////////////////////////////////////////////////////////////////
