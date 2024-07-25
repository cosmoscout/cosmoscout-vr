////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "advanced_modes.cuh"

#include "atmosphere_rendering.cuh"
#include "common.hpp"
#include "gpuErrCheck.hpp"
#include "math.cuh"

#include <stb_image_write.h>

#include <cstdint>
#include <iostream>

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace {

////////////////////////////////////////////////////////////////////////////////////////////////////

enum class Mode { eBruneton, ePlanetView, eAtmoView };

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

double __device__ getSunIlluminance(double sunDistance) {
  const double sunLuminousPower = 3.75e28;
  return sunLuminousPower / (4.0 * glm::pi<double>() * sunDistance * sunDistance);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void computeShadowMap(common::OutputSettings output, common::MappingSettings mapping,
    common::GeometrySettings geometry, common::LimbDarkening limbDarkening,
    advanced::Textures textures) {

  uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
  uint32_t i = y * output.mSize + x;

  if ((x >= output.mSize) || (y >= output.mSize)) {
    return;
  }

  uint32_t  samplesX = 64;
  uint32_t  samplesY = 64;
  glm::vec3 indirectIlluminance(0.0);

  double   phiOcc, phiSun, delta;
  uint32_t iterations = math::mapPixelToAngles(
      glm::ivec2(x, y), output.mSize, mapping, geometry, phiOcc, phiSun, delta);

  if (i == 100) {
    printf("phiSun: %f\n", phiSun);
    printf("phiOcc: %f\n", phiOcc);
    printf("delta: %f\n", delta);
    printf("iterations: %d\n", iterations);
  }

  double occDist    = geometry.mRadiusOcc / glm::sin(phiOcc);
  double sunDist    = geometry.mRadiusSun / glm::sin(phiSun);
  double atmoRadius = geometry.mRadiusAtmo;
  double phiAtmo    = glm::asin(atmoRadius / occDist);

  glm::dvec3 camera       = glm::dvec3(0.0, 0.0, occDist);
  glm::dvec3 sunDirection = glm::dvec3(0.0, glm::sin(delta), -glm::cos(delta));

  // Compute the direction of the ray.

  for (uint32_t sampleY = 0; sampleY < samplesY; ++sampleY) {
    double altitude         = ((double)sampleY + 0.5) / samplesY;
    double upperAltitude    = ((double)sampleY + 1.0) / samplesY;
    double lowerAltitude    = ((double)sampleY) / samplesY;
    double upperPhiRay      = phiOcc + upperAltitude * (phiAtmo - phiOcc);
    double lowerPhiRay      = phiOcc + lowerAltitude * (phiAtmo - phiOcc);
    double rowSolidAngle    = 0.5 * (math::getCapArea(upperPhiRay) - math::getCapArea(lowerPhiRay));
    double sampleSolidAngle = rowSolidAngle / samplesX;

    for (uint32_t sampleX = 0; sampleX < samplesX; ++sampleX) {

      double theta = (((double)sampleX + 0.5) / samplesX) * M_PI;

      double     phiRay = phiOcc + altitude * (phiAtmo - phiOcc);
      glm::dvec3 rayDir = glm::dvec3(0.0, glm::sin(phiRay), -glm::cos(phiRay));
      rayDir =
          glm::normalize(math::rotateVector(rayDir, glm::dvec3(0.0, 0.0, -1.0), glm::cos(theta)));

      glm::vec3 luminance = advanced::getLuminance(
          camera, rayDir, sunDirection, geometry, limbDarkening, textures, phiSun);

      indirectIlluminance += luminance * (float)(sampleSolidAngle / getSunIlluminance(sunDist));
    }
  }

  double sunArea = math::getCircleArea(1.0);
  double radiusOcc, distance;
  math::mapPixelToRadii(glm::ivec2(x, y), output.mSize, mapping, radiusOcc, distance);
  double directIlluminance =
      1.0 - math::sampleCircleIntersection(1.0,
                radiusOcc * geometry.mRadiusAtmo / geometry.mRadiusOcc, distance, limbDarkening) /
                sunArea;

  output.mBuffer[i * 3 + 0] = indirectIlluminance.r + directIlluminance;
  output.mBuffer[i * 3 + 1] = indirectIlluminance.g + directIlluminance;
  output.mBuffer[i * 3 + 2] = indirectIlluminance.b + directIlluminance;
}
////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void drawAtmoView(common::MappingSettings mapping, common::GeometrySettings geometry,
    float exposure, double phiOcc, double phiSun, double delta, common::OutputSettings output,
    common::LimbDarkening limbDarkening, advanced::Textures textures) {

  uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
  uint32_t i = y * output.mSize + x;

  if ((x >= output.mSize) || (y >= output.mSize)) {
    return;
  }

  double occDist    = geometry.mRadiusOcc / glm::sin(phiOcc);
  double atmoRadius = geometry.mRadiusAtmo;
  double phiAtmo    = glm::asin(atmoRadius / occDist);

  glm::dvec3 camera       = glm::dvec3(0.0, 0.0, occDist);
  glm::dvec3 sunDirection = glm::dvec3(0.0, glm::sin(delta), -glm::cos(delta));

  // Compute the direction of the ray.
  double theta    = (x / (double)output.mSize) * M_PI;
  double altitude = (y / (double)output.mSize);

  double     phiRay = phiOcc + altitude * (phiAtmo - phiOcc);
  glm::dvec3 rayDir = glm::dvec3(0.0, glm::sin(phiRay), -glm::cos(phiRay));
  rayDir = glm::normalize(math::rotateVector(rayDir, glm::dvec3(0.0, 0.0, -1.0), glm::cos(theta)));

  glm::vec3 luminance = advanced::getLuminance(
      camera, rayDir, sunDirection, geometry, limbDarkening, textures, phiSun);

  luminance = linearToSRGB(tonemap(luminance * exposure));

  output.mBuffer[i * 3 + 0] = luminance.r;
  output.mBuffer[i * 3 + 1] = luminance.g;
  output.mBuffer[i * 3 + 2] = luminance.b;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void drawPlanet(common::MappingSettings mapping, common::GeometrySettings geometry,
    float exposure, double phiOcc, double phiSun, double delta, float fov,
    common::OutputSettings output, common::LimbDarkening limbDarkening,
    advanced::Textures textures) {

  uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
  uint32_t i = y * output.mSize + x;

  if ((x >= output.mSize) || (y >= output.mSize)) {
    return;
  }

  // Total eclipse from Moon, horizon close up.
  double     occDist      = geometry.mRadiusOcc / glm::sin(phiOcc);
  glm::dvec3 camera       = glm::dvec3(0.0, 0.0, occDist);
  double     fieldOfView  = fov * M_PI / 180.0;
  glm::dvec3 sunDirection = glm::dvec3(0.0, glm::sin(delta), -glm::cos(delta));

  // Compute the direction of the ray.
  double theta = (x / (double)output.mSize - 0.5) * fieldOfView;
  double phi   = (y / (double)output.mSize - 0.5) * fieldOfView;

  glm::dvec3 rayDir =
      glm::dvec3(glm::sin(theta) * glm::cos(phi), glm::sin(phi), -glm::cos(theta) * glm::cos(phi));

  glm::vec3 luminance = advanced::getLuminance(
      camera, rayDir, sunDirection, geometry, limbDarkening, textures, phiSun);

  luminance = linearToSRGB(tonemap(luminance * exposure));

  output.mBuffer[i * 3 + 0] = luminance.r;
  output.mBuffer[i * 3 + 1] = luminance.g;
  output.mBuffer[i * 3 + 2] = luminance.b;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

int run(Mode mode, std::vector<std::string> const& arguments) {

  std::string              input;
  common::MappingSettings  mapping;
  common::OutputSettings   output;
  common::GeometrySettings geometry;
  bool                     printHelp = false;

  // These are only required for the planet or atmosphere view modes.
  float exposure = 0.0001; // The exposure of the image used during tonemapping.
  float x        = 0.5;    // The shadow map x coordinate for which to render the view.
  float y        = 0.5;    // The shadow map y coordinate for which to render the view.

  // This is only required for the planet view mode.
  float fov = 45.0; // The field of view of the camera in degrees.

  // First configure all possible command line options.
  cs::utils::CommandLine args("Here are the available options:");
  common::addMappingSettingsFlags(args, mapping);
  common::addOutputSettingsFlags(args, output);
  common::addGeometrySettingsFlags(args, geometry);

  args.addArgument({"--input"}, &input, "The path to the atmosphere settings directory.");

  if (mode == Mode::eAtmoView || mode == Mode::ePlanetView) {
    args.addArgument({"--exposure"}, &exposure,
        "The exposure of the image. Default is " + std::to_string(exposure));
    args.addArgument({"--x"}, &x,
        "The shadow map x coordinate for which to render the view. "
        "Default is " +
            std::to_string(x));
    args.addArgument({"--y"}, &y,
        "The shadow map y coordinate for which to render the view. "
        "Default is " +
            std::to_string(y));
  }

  if (mode == Mode::ePlanetView) {
    args.addArgument({"--fov"}, &fov,
        "The field of view of the camera in degrees. Default is " + std::to_string(fov));
  }

  args.addArgument({"-h", "--help"}, &printHelp, "Show this help message.");

  // Then do the actual parsing.
  try {
    args.parse(arguments);
  } catch (std::runtime_error const& e) {
    std::cerr << "Failed to parse command line arguments: " << e.what() << std::endl;
    return 1;
  }

  // When printHelp was set to true, we print a help message and exit.
  if (printHelp) {
    args.printHelp();
    return 0;
  }

  // If we are in atmosphere mode, we need also the atmosphere settings.
  if (input.empty()) {
    std::cerr << "When using the 'bruneton', 'planet-view', or 'atmo-view' mode, you must provide "
                 "the path to the atmosphere settings directory using --input!"
              << std::endl;
    return 1;
  }

  // Load the atmosphere settings.
  auto textures = advanced::loadTextures(input);

  // Initialize the limb darkening model.
  common::LimbDarkening limbDarkening;
  limbDarkening.init();

  // Compute the 2D kernel size.
  dim3     blockSize(16, 16);
  uint32_t numBlocksX = (output.mSize + blockSize.x - 1) / blockSize.x;
  uint32_t numBlocksY = (output.mSize + blockSize.y - 1) / blockSize.y;
  dim3     gridSize   = dim3(numBlocksX, numBlocksY);

  // Allocate the shared memory for the shadow map.
  gpuErrchk(cudaMallocManaged(
      &output.mBuffer, static_cast<size_t>(output.mSize * output.mSize) * 3 * sizeof(float)));

  if (mode == Mode::eBruneton) {
    computeShadowMap<<<gridSize, blockSize>>>(output, mapping, geometry, limbDarkening, textures);
  } else {

    double   phiOcc, phiSun, delta;
    uint32_t iterations = math::mapPixelToAngles(glm::ivec2(x * output.mSize, y * output.mSize),
        output.mSize, mapping, geometry, phiOcc, phiSun, delta);

    std::cout << "Required " << iterations << " iterations to find the correct angles."
              << std::endl;

    if (mode == Mode::ePlanetView) {
      drawPlanet<<<gridSize, blockSize>>>(
          mapping, geometry, exposure, phiOcc, phiSun, delta, fov, output, limbDarkening, textures);
    } else if (mode == Mode::eAtmoView) {
      drawAtmoView<<<gridSize, blockSize>>>(
          mapping, geometry, exposure, phiOcc, phiSun, delta, output, limbDarkening, textures);
    }
  }

  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());

  // Finally write the output texture!
  stbi_write_hdr(output.mFile.c_str(), static_cast<int>(output.mSize),
      static_cast<int>(output.mSize), 3, output.mBuffer);

  // Free the shared memory.
  gpuErrchk(cudaFree(output.mBuffer));

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
