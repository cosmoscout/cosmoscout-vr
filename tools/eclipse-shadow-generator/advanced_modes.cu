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
#include "tiff_utils.hpp"

#include <cstdint>
#include <iostream>

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace {

////////////////////////////////////////////////////////////////////////////////////////////////////

enum class Mode { eShadow, eLimbLuminance, ePlanetView, eAtmoView };

////////////////////////////////////////////////////////////////////////////////////////////////////

// Tonemapping code and color space conversions.
// http://filmicworlds.com/blog/filmic-tonemapping-operators/

__device__ glm::vec3 uncharted2Tonemap(glm::vec3 color) {
  const float A = 0.15;
  const float B = 0.50;
  const float C = 0.10;
  const float D = 0.20;
  const float E = 0.02;
  const float F = 0.30;
  return ((color * (A * color + C * B) + D * E) / (color * (A * color + B) + D * F)) - E / F;
}

__device__ glm::vec3 tonemap(glm::vec3 color) {
  const float W        = 11.2;
  color                = uncharted2Tonemap(10.0f * color);
  glm::vec3 whiteScale = glm::vec3(1.0) / uncharted2Tonemap(glm::vec3(W));
  return color * whiteScale;
}

__device__ float linearToSRGB(float value) {
  if (value <= 0.0031308f)
    return 12.92f * value;
  else
    return 1.055f * pow(value, 1.0f / 2.4f) - 0.055f;
}

__device__ glm::vec3 linearToSRGB(glm::vec3 color) {
  return glm::vec3(linearToSRGB(color.r), linearToSRGB(color.g), linearToSRGB(color.b));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

double __device__ getSunIlluminance(double sunDistance) {
  const double sunLuminousPower = 3.75e28;
  return sunLuminousPower / (4.0 * glm::pi<double>() * sunDistance * sunDistance);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void computeShadowMap(common::Output output, common::Mapping mapping,
    common::Geometry geometry, common::LimbDarkening limbDarkening, advanced::Textures textures) {

  uint32_t uShadow = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t vShadow = blockIdx.y * blockDim.y + threadIdx.y;
  uint32_t i       = vShadow * output.mSize + uShadow;

  if ((uShadow >= output.mSize) || (vShadow >= output.mSize)) {
    return;
  }

  // For integrating the luminance over all directions, we render an image of the atmosphere from
  // the perspective of the point in space. We use a parametrization of the texture space which
  // contains exactly on half of the atmosphere as seen from the point. The individual sample points
  // are weighted by the solid angle they cover on the sphere around the point.
  //
  //        ┌---..
  //  vLimb │      '
  //        └--.     \
  //      uLimb \     │
  //             │    │
  //            /     │
  //        ┌--'     /
  //        │      .
  //        └---''
  //
  // We use this resolution for the integration:
  uint32_t samplesULimb = 256;
  uint32_t samplesVLimb = 256;

  // First, compute the angular radii of Sun and occluder as well as the angle between the two.
  double phiOcc, phiSun, delta;
  math::mapPixelToAngles(
      glm::ivec2(uShadow, vShadow), output.mSize, mapping, geometry, phiOcc, phiSun, delta);

  // Make sure to stick to positions outside the atmosphere.
  double occDist    = glm::max(geometry.mRadiusOcc / glm::sin(phiOcc), geometry.mRadiusAtmo);
  double sunDist    = geometry.mRadiusSun / glm::sin(phiSun);
  double atmoRadius = geometry.mRadiusAtmo;
  double phiAtmo    = glm::asin(atmoRadius / occDist);

  glm::dvec3 camera       = glm::dvec3(0.0, 0.0, occDist);
  glm::dvec3 sunDirection = glm::dvec3(0.0, glm::sin(delta), -glm::cos(delta));

  glm::vec3 indirectIlluminance(0.0);

  for (uint32_t sampleV = 0; sampleV < samplesVLimb; ++sampleV) {
    double vLimb            = (static_cast<double>(sampleV) + 0.5) / samplesVLimb;
    double upperBound       = (static_cast<double>(sampleV) + 1.0) / samplesVLimb;
    double lowerBound       = static_cast<double>(sampleV) / samplesVLimb;
    double upperPhiRay      = phiOcc + upperBound * (phiAtmo - phiOcc);
    double lowerPhiRay      = phiOcc + lowerBound * (phiAtmo - phiOcc);
    double rowSolidAngle    = 0.5 * (math::getCapArea(upperPhiRay) - math::getCapArea(lowerPhiRay));
    double sampleSolidAngle = rowSolidAngle / samplesULimb;

    for (uint32_t sampleU = 0; sampleU < samplesULimb; ++sampleU) {

      double beta = ((static_cast<double>(sampleU) + 0.5) / samplesULimb) * M_PI;

      // Compute the direction of the ray.
      double     phiRay = phiOcc + vLimb * (phiAtmo - phiOcc);
      glm::dvec3 rayDir = glm::dvec3(0.0, glm::sin(phiRay), -glm::cos(phiRay));
      rayDir =
          glm::normalize(math::rotateVector(rayDir, glm::dvec3(0.0, 0.0, -1.0), glm::cos(beta)));

      glm::vec3 luminance = advanced::getLuminance(
          camera, rayDir, sunDirection, geometry, limbDarkening, textures, phiSun);

      indirectIlluminance += luminance * static_cast<float>(sampleSolidAngle);
    }
  }

  // We only computed half of the atmosphere, so we multiply the result by two.
  indirectIlluminance *= 2.0;

  // We now have the light which reaches our point through the atmosphere. However, there is also a
  // certain amount of direct sunlight which reaches the point from paths which do not intersect the
  // atmosphere. We use the formula from the simple mode to compute the visible fraction of the Sun
  // above the upper atmosphere boundary.

  double sunArea = math::getCircleArea(1.0);
  double radiusOcc, distance;
  math::mapPixelToRadii(glm::ivec2(uShadow, vShadow), output.mSize, mapping, radiusOcc, distance);
  double radiusAtmo = radiusOcc * geometry.mRadiusAtmo / geometry.mRadiusOcc;
  double visibleFraction =
      1.0 - math::sampleCircleIntersection(1.0, radiusAtmo, distance, limbDarkening) / sunArea;

  // We multiply this fraction with the illuminance of the Sun to get the direct
  // illuminance.
  double fullIlluminance   = getSunIlluminance(sunDist);
  double directIlluminance = fullIlluminance * visibleFraction;

  // We add the direct and indirect illuminance to get the total illuminance.
  glm::dvec3 totalIlluminance = indirectIlluminance + glm::vec3(directIlluminance);

  // And divide by the illuminance at the point if there were no atmosphere and no planet.
  output.mBuffer[i * 3 + 0] = totalIlluminance.r / fullIlluminance;
  output.mBuffer[i * 3 + 1] = totalIlluminance.g / fullIlluminance;
  output.mBuffer[i * 3 + 2] = totalIlluminance.b / fullIlluminance;

  // Print a rough progress estimate.
  if (i % 1000 == 0) {
    printf("Progress: %f%%\n", (i / static_cast<float>(output.mSize * output.mSize)) * 100.0);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void computeLimbLuminance(common::Output output, common::Mapping mapping,
    common::Geometry geometry, common::LimbDarkening limbDarkening, advanced::Textures textures,
    int layers) {

  uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
  uint32_t z = blockIdx.z * blockDim.z + threadIdx.z;
  uint32_t i = z * output.mSize * output.mSize + y * output.mSize + x;

  if ((x >= output.mSize) || (y >= output.mSize) || (z >= output.mSize * layers)) {
    return;
  }

  // For precomputing the atmosphere's luminance for every position in the shadow volume, we render
  // an image of the atmosphere from the perspective of the point in space. We use a parametrization
  // of the texture space which contains exactly on half of the atmosphere as seen from the point.
  // We render with a relatively high resolution in the vertical direction to capture even a small
  // refracted image of the Sun. The output texture however only contain a few layers in the vLimb
  // direction to keep the memory requirements low. For this, we render each layer separately and
  // integrate the luminance over the vLimb direction.
  //
  //            uLimb - .
  //       ^   ┌---..     '
  // vLimb │   ├ -.  /'     \ 
  //       │   └--./ .  \    │
  //           │β⁠/ \  .  │   V
  //           o    │ .  │
  //               /  .  │
  //           ┌--'  .  /
  //           ├ - '   .
  //           └---''
  //
  // The output texture is a 4D texture stored in a 3D texture: The x and y coordinates are the
  // usual shadow map coordinates, and the z coordinate contains the layers of the atmosphere image
  // around the planet. The resolution of the texture is [output.mSize, output.mSize, output.mSize *
  // layers].
  uint32_t samplesULimb = output.mSize;

  // We use this many vertical samples for each layer.
  uint32_t samplesVLimb = 256;

  // This thread computes the luminance for this layer and this position along the atmosphere ring.
  uint32_t layer   = z / output.mSize;
  uint32_t sampleU = z % output.mSize;

  // First, compute the angular radii of Sun and occluder as well as the angle between the two.
  double phiOcc, phiSun, delta;
  math::mapPixelToAngles(glm::ivec2(x, y), output.mSize, mapping, geometry, phiOcc, phiSun, delta);

  // Make sure to stick to positions outside the atmosphere.
  double occDist    = glm::max(geometry.mRadiusOcc / glm::sin(phiOcc), geometry.mRadiusAtmo);
  double sunDist    = geometry.mRadiusSun / glm::sin(phiSun);
  double atmoRadius = geometry.mRadiusAtmo;
  double phiAtmo    = glm::asin(atmoRadius / occDist);

  glm::dvec3 camera       = glm::dvec3(0.0, 0.0, occDist);
  glm::dvec3 sunDirection = glm::dvec3(0.0, glm::sin(delta), -glm::cos(delta));

  double beta = ((static_cast<double>(sampleU) + 0.5) / samplesULimb) * M_PI;

  glm::vec3 luminance(0.0);

  for (uint32_t sampleV = 0; sampleV < samplesVLimb; ++sampleV) {
    double vLimb = (static_cast<double>(sampleV) + 0.5) / samplesVLimb;

    double layerStart = static_cast<double>(layer) / layers;
    double layerEnd   = static_cast<double>(layer + 1) / layers;
    vLimb             = layerStart + vLimb * (layerEnd - layerStart);

    // Compute the direction of the ray.
    double     phiRay = phiOcc + vLimb * (phiAtmo - phiOcc);
    glm::dvec3 rayDir = glm::dvec3(0.0, glm::sin(phiRay), -glm::cos(phiRay));
    rayDir = glm::normalize(math::rotateVector(rayDir, glm::dvec3(0.0, 0.0, -1.0), glm::cos(beta)));

    luminance += advanced::getLuminance(
                     camera, rayDir, sunDirection, geometry, limbDarkening, textures, phiSun) /
                 static_cast<float>(samplesVLimb);
  }

  // And divide by the illuminance at the point if there were no atmosphere and no planet.
  output.mBuffer[i * 3 + 0] = luminance.r;
  output.mBuffer[i * 3 + 1] = luminance.g;
  output.mBuffer[i * 3 + 2] = luminance.b;

  // Print a rough progress estimate.
  if (i % 1000 == 0) {
    printf("Progress: %f%%\n",
        (i / static_cast<float>(output.mSize * output.mSize * output.mSize * layers)) * 100.0);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void drawAtmoView(common::Mapping mapping, common::Geometry geometry, float exposure,
    double phiOcc, double phiSun, double delta, common::Output output,
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
  double beta  = (x / static_cast<double>(output.mSize)) * M_PI;
  double vLimb = (y / static_cast<double>(output.mSize));

  double     phiRay = phiOcc + vLimb * (phiAtmo - phiOcc);
  glm::dvec3 rayDir = glm::dvec3(0.0, glm::sin(phiRay), -glm::cos(phiRay));
  rayDir = glm::normalize(math::rotateVector(rayDir, glm::dvec3(0.0, 0.0, -1.0), glm::cos(beta)));

  glm::vec3 luminance = advanced::getLuminance(
      camera, rayDir, sunDirection, geometry, limbDarkening, textures, phiSun);

  luminance = linearToSRGB(tonemap(luminance * exposure));

  output.mBuffer[i * 3 + 0] = luminance.r;
  output.mBuffer[i * 3 + 1] = luminance.g;
  output.mBuffer[i * 3 + 2] = luminance.b;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void drawPlanet(common::Mapping mapping, common::Geometry geometry, float exposure,
    double phiOcc, double phiSun, double delta, float fov, common::Output output,
    common::LimbDarkening limbDarkening, advanced::Textures textures) {

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
  double theta = (x / static_cast<double>(output.mSize) - 0.5) * fieldOfView;
  double phi   = (y / static_cast<double>(output.mSize) - 0.5) * fieldOfView;

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

  std::string      input;
  common::Mapping  mapping;
  common::Output   output;
  common::Geometry geometry;
  bool             printHelp = false;

  // These are only required for the planet or atmosphere view modes.
  float exposure = 0.0001; // The exposure of the image used during tonemapping.
  float x        = 0.5;    // The shadow map x coordinate for which to render the view.
  float y        = 0.5;    // The shadow map y coordinate for which to render the view.

  // This is only required for the planet view mode.
  float fov = 45.0; // The field of view of the camera in degrees.

  // This is only required for the limb luminance mode.
  int limbLuminanceLayers = 1; // The number of vertical layers in the limb luminance texture.

  // First configure all possible command line options.
  cs::utils::CommandLine args("Here are the available options:");
  common::addMappingFlags(args, mapping);
  common::addOutputFlags(args, output);
  common::addGeometryFlags(args, geometry);

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

  if (mode == Mode::eLimbLuminance) {
    args.addArgument({"--layers"}, &limbLuminanceLayers,
        "The number of vertical layers in the limb luminance texture. Default is " +
            std::to_string(limbLuminanceLayers));
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
  dim3     blockSize(8, 8, mode == Mode::eLimbLuminance ? 8 : 1);
  uint32_t numBlocksX = (output.mSize + blockSize.x - 1) / blockSize.x;
  uint32_t numBlocksY = (output.mSize + blockSize.y - 1) / blockSize.y;
  uint32_t numBlocksZ = mode == Mode::eLimbLuminance
                            ? (output.mSize * limbLuminanceLayers + blockSize.z - 1) / blockSize.z
                            : 1;
  dim3     gridSize   = dim3(numBlocksX, numBlocksY, numBlocksZ);

  // Allocate the shared memory for the shadow map.
  if (mode == Mode::eLimbLuminance) {
    gpuErrchk(cudaMallocManaged(&output.mBuffer,
        static_cast<size_t>(output.mSize * output.mSize * output.mSize * limbLuminanceLayers) * 3 *
            sizeof(float)));
  } else {
    gpuErrchk(cudaMallocManaged(
        &output.mBuffer, static_cast<size_t>(output.mSize * output.mSize) * 3 * sizeof(float)));
  }

  if (mode == Mode::eShadow) {
    computeShadowMap<<<gridSize, blockSize>>>(output, mapping, geometry, limbDarkening, textures);
  } else if (mode == Mode::eLimbLuminance) {
    computeLimbLuminance<<<gridSize, blockSize>>>(
        output, mapping, geometry, limbDarkening, textures, limbLuminanceLayers);
  } else {

    double     phiOcc, phiSun, delta;
    glm::ivec2 pixel(x * output.mSize, y * output.mSize);
    uint32_t   iterations =
        math::mapPixelToAngles(pixel, output.mSize, mapping, geometry, phiOcc, phiSun, delta);

    if (iterations == 0) {
      std::cerr << "The given pixel is in an impossible configuration." << std::endl;
      return 1;
    }

    std::cout << "Required " << iterations << " iterations to find the correct angles for pixel ("
              << pixel.x << ", " << pixel.y << ")." << std::endl;
    std::cout << " - Observer Angular Radius: " << glm::degrees(phiOcc) << "°" << std::endl;
    std::cout << " - Observer Distance: " << geometry.mRadiusOcc / glm::sin(phiOcc) * 0.001 << " km"
              << std::endl;
    std::cout << " - Sun Angular Radius: " << glm::degrees(phiSun) << "°" << std::endl;
    std::cout << " - Sun Elevation: " << glm::degrees(delta) << "°" << std::endl;

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
  if (mode == Mode::eLimbLuminance) {
    tiff_utils::write3D(output.mFile, output.mBuffer, static_cast<int>(output.mSize),
        static_cast<int>(output.mSize), static_cast<int>(output.mSize * limbLuminanceLayers), 3);
  } else {
    tiff_utils::write2D(output.mFile, output.mBuffer, static_cast<int>(output.mSize),
        static_cast<int>(output.mSize), 3);
  }

  // Free the shared memory.
  gpuErrchk(cudaFree(output.mBuffer));

  return 0;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace advanced {

////////////////////////////////////////////////////////////////////////////////////////////////////

int shadowMode(std::vector<std::string> const& arguments) {
  return run(Mode::eShadow, arguments);
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

int limbLuminanceMode(std::vector<std::string> const& arguments) {
  return run(Mode::eLimbLuminance, arguments);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace advanced

////////////////////////////////////////////////////////////////////////////////////////////////////
