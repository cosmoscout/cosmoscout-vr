////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-FileCopyrightText: 2017 Eric Bruneton
// SPDX-FileCopyrightText: 2008 INRIA
// SPDX-License-Identifier: BSD-3-Clause

// Parts of this file are based on the original implementation by Eric Bruneton:
// https://github.com/ebruneton/precomputed_atmospheric_scattering/blob/master/atmosphere/functions.glsl

// It has been ported to CUDA and in some cases it has been simplified as we are only interested in
// vantage points from outer space.

// All methods which are based on the original implementation by Eric Bruneton are marked with a
// corresponding comment and a link to the original source code.

#include "atmosphere_rendering.cuh"

#include "tiff_utils.hpp"

#include <fstream>
#include <nlohmann/json.hpp>

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace {

////////////////////////////////////////////////////////////////////////////////////////////////////

// Returns the luminance of the Sun in candela per square meter.
double __device__ getSunLuminance(double sunRadius) {
  const double sunLuminousPower = 3.75e28;
  const double sunLuminousExitance =
      sunLuminousPower / (sunRadius * sunRadius * 4.0 * glm::pi<double>());
  return sunLuminousExitance / glm::pi<double>();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Creates a CUDA texture object from the given RGBA texture.
cudaTextureObject_t createCudaTexture(tiff_utils::RGBATexture const& texture) {
  cudaArray* cuArray;
  auto       channelDesc = cudaCreateChannelDesc<float4>();

  cudaMallocArray(&cuArray, &channelDesc, texture.width, texture.height);
  cudaMemcpy2DToArray(cuArray, 0, 0, texture.data.data(), texture.width * sizeof(float) * 4,
      texture.width * sizeof(float) * 4, texture.height, cudaMemcpyHostToDevice);

  cudaResourceDesc resDesc = {};
  resDesc.resType          = cudaResourceTypeArray;
  resDesc.res.array.array  = cuArray;

  cudaTextureDesc texDesc  = {};
  texDesc.addressMode[0]   = cudaAddressModeClamp;
  texDesc.addressMode[1]   = cudaAddressModeClamp;
  texDesc.filterMode       = cudaFilterModeLinear;
  texDesc.readMode         = cudaReadModeElementType;
  texDesc.normalizedCoords = 1;

  cudaTextureObject_t textureObject = 0;
  cudaCreateTextureObject(&textureObject, &resDesc, &texDesc, nullptr);

  gpuErrchk(cudaGetLastError());

  return textureObject;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Provides a similar API to the texture2D function in GLSL. Returns the RGBA value at the given
// texture coordinates as a glm::vec4.
__device__ glm::vec4 texture2D(cudaTextureObject_t tex, glm::vec2 uv) {
  auto data = tex2D<float4>(tex, uv.x, uv.y);
  return glm::vec4(data.x, data.y, data.z, data.w);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// In case the input value is negative, this function returns 0.0. Otherwise it returns the square
// root of the input value.
__device__ float safeSqrt(float a) {
  return glm::sqrt(glm::max(a, 0.0f));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// https://github.com/ebruneton/precomputed_atmospheric_scattering/blob/master/atmosphere/functions.glsl#L342
__device__ float getTextureCoordFromUnitRange(float x, int textureSize) {
  return 0.5 / float(textureSize) + x * (1.0 - 1.0 / float(textureSize));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// https://github.com/ebruneton/precomputed_atmospheric_scattering/blob/master/atmosphere/functions.glsl#L207
__device__ float distanceToTopAtmosphereBoundary(
    common::Geometry const& geometry, float r, float mu) {
  float discriminant = r * r * (mu * mu - 1.0) + geometry.mRadiusAtmo * geometry.mRadiusAtmo;
  return glm::max(0.f, -r * mu + safeSqrt(discriminant));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// As we are always in outer space, this function does not need the r parameter when compared to the
// original version:
// https://github.com/ebruneton/precomputed_atmospheric_scattering/blob/master/atmosphere/functions.glsl#L402
__device__ glm::vec2 getTransmittanceTextureUvFromRMu(
    advanced::Textures const& textures, common::Geometry const& geometry, double mu) {

  // Distance to top atmosphere boundary for a horizontal ray at ground level.
  double H =
      sqrt(geometry.mRadiusAtmo * geometry.mRadiusAtmo - geometry.mRadiusOcc * geometry.mRadiusOcc);

  // Distance to the top atmosphere boundary for the ray (r,mu), and its minimum
  // and maximum values over all mu - obtained for (r,1) and (r,mu_horizon).
  double d    = distanceToTopAtmosphereBoundary(geometry, geometry.mRadiusAtmo, mu);
  double dMax = 2.0 * H;
  double xMu  = d / dMax;
  return glm::vec2(getTextureCoordFromUnitRange(xMu, textures.mTransmittanceTextureWidth),
      getTextureCoordFromUnitRange(1.0, textures.mTransmittanceTextureHeight));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// As we are always in outer space, this function does not need the r parameter when compared to the
// original version:
// https://github.com/ebruneton/precomputed_atmospheric_scattering/blob/master/atmosphere/functions.glsl#L773
__device__ glm::vec3 getScatteringTextureUvwFromRMuMuSNu(advanced::Textures const& textures,
    common::Geometry const& geometry, double mu, double muS, double nu,
    bool rayRMuIntersectsGround) {

  // Distance to top atmosphere boundary for a horizontal ray at ground level.
  double H =
      sqrt(geometry.mRadiusAtmo * geometry.mRadiusAtmo - geometry.mRadiusOcc * geometry.mRadiusOcc);

  // Discriminant of the quadratic equation for the intersections of the ray (r,mu) with the ground
  // (see rayIntersectsGround).
  double rMu          = geometry.mRadiusAtmo * mu;
  double discriminant = rMu * rMu - geometry.mRadiusAtmo * geometry.mRadiusAtmo +
                        geometry.mRadiusOcc * geometry.mRadiusOcc;
  double uMu;
  if (rayRMuIntersectsGround) {
    // Distance to the ground for the ray (r,mu), and its minimum and maximum values over all mu -
    // obtained for (r,-1) and (r,mu_horizon).
    double d    = -rMu - safeSqrt(discriminant);
    double dMin = geometry.mRadiusAtmo - geometry.mRadiusOcc;
    double dMax = H;
    uMu = 0.5 - 0.5 * getTextureCoordFromUnitRange(dMax == dMin ? 0.0 : (d - dMin) / (dMax - dMin),
                          textures.mScatteringTextureMuSize / 2);
  } else {
    // Distance to the top atmosphere boundary for the ray (r,mu), and its minimum and maximum
    // values over all mu - obtained for (r,1) and (r,mu_horizon).
    double d    = -rMu + safeSqrt(discriminant + H * H);
    double dMax = 2.0 * H;
    uMu = 0.5 + 0.5 * getTextureCoordFromUnitRange(d / dMax, textures.mScatteringTextureMuSize / 2);
  }

  double d    = distanceToTopAtmosphereBoundary(geometry, geometry.mRadiusOcc, muS);
  double dMin = geometry.mRadiusAtmo - geometry.mRadiusOcc;
  double dMax = H;
  double a    = (d - dMin) / (dMax - dMin);
  double D    = distanceToTopAtmosphereBoundary(geometry, geometry.mRadiusOcc, textures.mMuSMin);
  double A    = (D - dMin) / (dMax - dMin);
  // An ad-hoc function equal to 0 for muS = MU_S_MIN (because then d = D and thus a = A), equal to
  // 1 for muS = 1 (because then d = dMin and thus a = 0), and with a large slope around muS = 0,
  // to get more texture samples near the horizon.
  float uMuS = getTextureCoordFromUnitRange(
      glm::max(1.0 - a / A, 0.0) / (1.0 + a), textures.mScatteringTextureMuSSize);

  float uNu = (nu + 1.0) / 2.0;
  return glm::vec3(uNu, uMuS, uMu);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// https://github.com/ebruneton/precomputed_atmospheric_scattering/blob/master/atmosphere/functions.glsl#L473
__device__ glm::vec3 getTransmittanceToTopAtmosphereBoundary(
    advanced::Textures const& textures, common::Geometry const& geometry, double mu) {
  glm::vec2 uv = getTransmittanceTextureUvFromRMu(textures, geometry, mu);
  return glm::vec3(texture2D(textures.mTransmittance, uv));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// https://github.com/ebruneton/precomputed_atmospheric_scattering/blob/master/atmosphere/functions.glsl#L240
__device__ bool rayIntersectsGround(common::Geometry const& geometry, double mu) {
  return mu < 0.0 && geometry.mRadiusAtmo * geometry.mRadiusAtmo * (mu * mu - 1.0) +
                             geometry.mRadiusOcc * geometry.mRadiusOcc >=
                         0.0;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// This is different. In the original implementation, the phase function is the Rayleigh phase
// function. We load the phase function from a texture.
__device__ glm::vec3 moleculePhaseFunction(cudaTextureObject_t phaseTexture, float nu) {
  float theta = glm::acos(nu) / M_PI; // 0<->1
  return glm::vec3(texture2D(phaseTexture, glm::vec2(theta, 0.0)));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// This is different. In the original implementation, the phase function is the Cornette-Shanks
// phase function. We load the phase function from a texture.
__device__ glm::vec3 aerosolPhaseFunction(cudaTextureObject_t phaseTexture, float nu) {
  float theta = glm::acos(nu) / M_PI; // 0<->1
  return glm::vec3(texture2D(phaseTexture, glm::vec2(theta, 1.0)));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// As we are always in outer space, this function does not need the r parameter when compared to the
// original version:
// https://github.com/ebruneton/precomputed_atmospheric_scattering/blob/master/atmosphere/functions.glsl#L1658
__device__ void getCombinedScattering(advanced::Textures const& textures,
    common::Geometry const& geometry, float mu, float muS, float nu, bool rayRMuIntersectsGround,
    glm::vec3& multipleScattering, glm::vec3& singleAerosolsScattering) {
  glm::vec3 uvw =
      getScatteringTextureUvwFromRMuMuSNu(textures, geometry, mu, muS, nu, rayRMuIntersectsGround);
  float     texCoordX = uvw.x * float(textures.mScatteringTextureNuSize - 1);
  float     texX      = floor(texCoordX);
  float     lerp      = texCoordX - texX;
  glm::vec2 uv0       = glm::vec2((texX + uvw.y) / float(textures.mScatteringTextureNuSize), uvw.z);
  glm::vec2 uv1 = glm::vec2((texX + 1.0 + uvw.y) / float(textures.mScatteringTextureNuSize), uvw.z);

  multipleScattering = glm::vec3(texture2D(textures.mMultipleScattering, uv0) * (1.0f - lerp) +
                                 texture2D(textures.mMultipleScattering, uv1) * lerp);
  singleAerosolsScattering =
      glm::vec3(texture2D(textures.mSingleAerosolsScattering, uv0) * (1.0f - lerp) +
                texture2D(textures.mSingleAerosolsScattering, uv1) * lerp);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Rotates the given ray towards the planet's surface by the angle given in the theta deviation
// texture. Also returns the contact radius which is the distance of closest approach to the
// planet's surface which the ray had when traveling through the atmosphere.
__device__ glm::dvec3 getRefractedRay(advanced::Textures const& textures,
    common::Geometry const& geometry, glm::dvec3 camera, glm::dvec3 ray, double& contactRadius) {

  // If refraction is disabled, we can simply return the ray. However, we still need to compute the
  // contact radius.
  if (!textures.mThetaDeviation) {
    double dist       = glm::length(camera);
    auto   toOccluder = -camera / dist;
    double angle      = math::angleBetweenVectors(ray, toOccluder);
    contactRadius     = glm::acos(angle) * dist - geometry.mRadiusOcc;

    return ray;
  }

  double    mu = dot(camera, ray) / geometry.mRadiusAtmo;
  glm::vec2 uv = getTransmittanceTextureUvFromRMu(textures, geometry, mu);

  // Cosine of the angular deviation of the ray due to refraction.
  glm::vec2  thetaDeviationContactRadius = glm::vec2(texture2D(textures.mThetaDeviation, uv));
  double     thetaDeviation              = glm::cos(double(thetaDeviationContactRadius.x));
  glm::dvec3 axis                        = glm::normalize(glm::cross(camera, ray));

  contactRadius = thetaDeviationContactRadius.y;

  return normalize(math::rotateVector(ray, axis, thetaDeviation));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace advanced {

////////////////////////////////////////////////////////////////////////////////////////////////////

// Loads all required textures from the given output directory from the Bruneton preprocessor tool.
__host__ Textures loadTextures(std::string const& path) {
  uint32_t scatteringTextureRSize = tiff_utils::getNumLayers(path + "/multiple_scattering.tif");

  tiff_utils::RGBATexture multiscattering =
      tiff_utils::read2DTexture(path + "/multiple_scattering.tif", scatteringTextureRSize - 1);
  tiff_utils::RGBATexture singleScattering = tiff_utils::read2DTexture(
      path + "/single_aerosols_scattering.tif", scatteringTextureRSize - 1);

  tiff_utils::RGBATexture phase         = tiff_utils::read2DTexture(path + "/phase.tif");
  tiff_utils::RGBATexture transmittance = tiff_utils::read2DTexture(path + "/transmittance.tif");

  Textures textures;
  textures.mMultipleScattering       = createCudaTexture(multiscattering);
  textures.mSingleAerosolsScattering = createCudaTexture(singleScattering);

  textures.mPhase         = createCudaTexture(phase);
  textures.mTransmittance = createCudaTexture(transmittance);

  std::ifstream  metaFile(path + "/metadata.json");
  nlohmann::json meta;
  metaFile >> meta;

  uint32_t scatteringTextureNuSize = meta.at("scatteringTextureNuSize");
  double   maxSunZenithAngle       = meta.at("maxSunZenithAngle");

  textures.mTransmittanceTextureWidth  = transmittance.width;
  textures.mTransmittanceTextureHeight = transmittance.height;
  textures.mScatteringTextureMuSize    = multiscattering.height;
  textures.mScatteringTextureMuSSize   = multiscattering.width / scatteringTextureNuSize;
  textures.mScatteringTextureNuSize    = scatteringTextureNuSize;
  textures.mMuSMin                     = std::cos(maxSunZenithAngle);

  bool enableRefraction = meta.at("refraction");
  if (enableRefraction) {
    tiff_utils::RGBATexture theta_deviation =
        tiff_utils::read2DTexture(path + "/theta_deviation.tif");
    textures.mThetaDeviation = createCudaTexture(theta_deviation);
  }

  return textures;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Computes the luminance of the atmosphere for the given geometry. All distances are in meters.
// This is loosely based on the original implementation by Eric Bruneton:
// https://github.com/ebruneton/precomputed_atmospheric_scattering/blob/master/atmosphere/functions.glsl#L1705
__device__ glm::vec3 getLuminance(glm::dvec3 camera, glm::dvec3 viewRay, glm::dvec3 sunDirection,
    common::Geometry const& geometry, common::LimbDarkening const& limbDarkening,
    Textures const& textures, double phiSun) {

  // Compute the distance to the top atmosphere boundary along the view ray, assuming the viewer is
  // in space (or NaN if the view ray does not intersect the atmosphere).
  double r   = length(camera);
  double rmu = dot(camera, viewRay);
  double distanceToTopAtmosphereBoundary =
      -rmu - sqrt(rmu * rmu - r * r + geometry.mRadiusAtmo * geometry.mRadiusAtmo);

  glm::vec3  skyLuminance  = glm::vec3(0.0);
  glm::vec3  transmittance = glm::vec3(1.0);
  glm::dvec3 refractedRay  = viewRay;

  // We only need to compute the luminance if the view ray intersects the atmosphere.
  if (distanceToTopAtmosphereBoundary > 0.0) {

    camera += viewRay * distanceToTopAtmosphereBoundary;

    // Compute the mu, muS and nu parameters needed for the texture lookups.
    double mu                     = (rmu + distanceToTopAtmosphereBoundary) / geometry.mRadiusAtmo;
    double muS                    = dot(camera, sunDirection) / geometry.mRadiusAtmo;
    double nu                     = dot(viewRay, sunDirection);
    bool   rayRMuIntersectsGround = rayIntersectsGround(geometry, mu);

    glm::vec3 multipleScattering;
    glm::vec3 singleAerosolsScattering;
    getCombinedScattering(textures, geometry, mu, muS, nu, rayRMuIntersectsGround,
        multipleScattering, singleAerosolsScattering);

    skyLuminance = multipleScattering * moleculePhaseFunction(textures.mPhase, nu) +
                   singleAerosolsScattering * aerosolPhaseFunction(textures.mPhase, nu);

    double contactRadius;
    refractedRay = getRefractedRay(textures, geometry, camera, viewRay, contactRadius);

    transmittance = rayRMuIntersectsGround
                        ? glm::vec3(0.0)
                        : getTransmittanceToTopAtmosphereBoundary(textures, geometry, mu);

    // To account for terrain height, we apply a very simple model. We assume that all rays passing
    // above two times the mean elevation of the terrain are not affected by the terrain. All rays
    // passing below zero elevation are completely blocked by the terrain. In between, the
    // transmittance is linearly interpolated. If a cloud elevation is set, all rays passing below
    // the cloud elevation are completely blocked by the cloud.
    if (contactRadius < geometry.mCloudAltitude) {
      transmittance = glm::vec3(0.0);
    } else if (contactRadius < 2.F * geometry.mAverageTerrainHeight) {
      float t = (2.F * geometry.mAverageTerrainHeight - contactRadius) /
                (2.f * geometry.mAverageTerrainHeight);
      transmittance = glm::mix(transmittance, glm::vec3(0.0), t);
    }
  }

  float sun = limbDarkening.get(math::angleBetweenVectors(refractedRay, sunDirection) / phiSun);

  glm::vec3 sunLuminance = transmittance * (float)getSunLuminance(geometry.mRadiusSun) * sun;

  return skyLuminance + sunLuminance;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace advanced

////////////////////////////////////////////////////////////////////////////////////////////////////
