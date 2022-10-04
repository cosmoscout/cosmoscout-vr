////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "../../src/cs-utils/CommandLine.hpp"

#include "LimbDarkening.cuh"
#include "math.cuh"

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image.h>
#include <stb_image_write.h>

////////////////////////////////////////////////////////////////////////////////////////////////////
// This tool can be used to create the eclipse shadow maps used by CosmoScout VR. See the         //
// README.md file in this directory for usage instructions!                                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

// This macro is used in multiple locations to check for Cuda errors.
// https://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
#define gpuErrchk(ans)                                                                             \
  { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) {
      exit(code);
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// This is used to pass the command line options to the Cuda kernel.
struct ShadowSettings {
  uint32_t size            = 512;
  bool     includeUmbra    = false;
  double   mappingExponent = 1.0;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

__constant__ LimbDarkening  cLimbDarkening;
__constant__ ShadowSettings cShadowSettings;

////////////////////////////////////////////////////////////////////////////////////////////////////

// Computes the shadow map by sampling the intersection area between circles representing the Sun
// and the occluder. This makes use of the global limb darkening function.
__global__ void computeLimbDarkeningShadow(float* shadowMap) {
  uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
  uint32_t i = y * cShadowSettings.size + x;

  if ((x >= cShadowSettings.size) || (y >= cShadowSettings.size)) {
    return;
  }

  auto angles = math::mapPixelToAngles(glm::ivec2(x, y), cShadowSettings.size,
      cShadowSettings.mappingExponent, cShadowSettings.includeUmbra);

  double sunArea = math::getCircleArea(1.0);

  shadowMap[i] = static_cast<float>(
      1 - math::sampleCircleIntersection(1.0, angles.x, angles.y, cLimbDarkening) / sunArea);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Computes the shadow map by analytically computing the intersection area between circles
// representing the Sun and the occluder. This does not use a limb darkening function.
__global__ void computeCircleIntersectionShadow(float* shadowMap) {
  uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
  uint32_t i = y * cShadowSettings.size + x;

  if ((x >= cShadowSettings.size) || (y >= cShadowSettings.size)) {
    return;
  }

  auto angles = math::mapPixelToAngles(glm::ivec2(x, y), cShadowSettings.size,
      cShadowSettings.mappingExponent, cShadowSettings.includeUmbra);

  double sunArea = math::getCircleArea(1.0);

  shadowMap[i] =
      static_cast<float>(1.0 - math::getCircleIntersection(1.0, angles.x, angles.y) / sunArea);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Computes the shadow map by assuming a linear brightness gradient from the outer edge of the
// penumbra to the start of the umbra / antumbra. In the antumbra, the shadow intensity decreases
// quadratically. This does not use a limb darkening function.
__global__ void computeLinearShadow(float* shadowMap) {
  uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
  uint32_t i = y * cShadowSettings.size + x;

  if ((x >= cShadowSettings.size) || (y >= cShadowSettings.size)) {
    return;
  }

  auto angles = math::mapPixelToAngles(glm::ivec2(x, y), cShadowSettings.size,
      cShadowSettings.mappingExponent, cShadowSettings.includeUmbra);

  double phiSun = 1.0;
  double phiOcc = angles[0];
  double delta  = angles[1];

  double visiblePortion =
      (delta - glm::abs(phiSun - phiOcc)) / (phiSun + phiOcc - glm::abs(phiSun - phiOcc));

  double maxDepth = glm::min(1.0, glm::pow(phiOcc / phiSun, 2.0));

  shadowMap[i] = static_cast<float>(1.0 - maxDepth * glm::clamp(1.0 - visiblePortion, 0.0, 1.0));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Computes the shadow map by assuming a smoothstep-based brightness gradient from the outer edge of
// the penumbra to the start of the umbra / antumbra. In the antumbra, the shadow intensity
// decreases quadratically. This does not use a limb darkening function.
__global__ void computeSmoothstepShadow(float* shadowMap) {
  uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
  uint32_t i = y * cShadowSettings.size + x;

  if ((x >= cShadowSettings.size) || (y >= cShadowSettings.size)) {
    return;
  }

  auto angles = math::mapPixelToAngles(glm::ivec2(x, y), cShadowSettings.size,
      cShadowSettings.mappingExponent, cShadowSettings.includeUmbra);

  double phiSun = 1.0;
  double phiOcc = angles[0];
  double delta  = angles[1];

  double visiblePortion =
      (delta - glm::abs(phiSun - phiOcc)) / (phiSun + phiOcc - glm::abs(phiSun - phiOcc));

  double maxDepth = glm::min(1.0, glm::pow(phiOcc / phiSun, 2.0));

  shadowMap[i] = static_cast<float>(
      1.0 - maxDepth * glm::clamp(1.0 - glm::smoothstep(0.0, 1.0, visiblePortion), 0.0, 1.0));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char** argv) {

  stbi_flip_vertically_on_write(1);

  ShadowSettings settings;

  std::string cOutput    = "shadow.hdr";
  std::string cMode      = "limb-darkening";
  bool        cPrintHelp = false;

  // First configure all possible command line options.
  cs::utils::CommandLine args(
      "Welcome to the shadow map generator! Here are the available options:");
  args.addArgument({"-o", "--output"}, &cOutput,
      "The image will be written to this file (default: \"" + cOutput + "\").");
  args.addArgument({"--size"}, &settings.size,
      "The output texture size (default: " + std::to_string(settings.size) + ").");
  args.addArgument({"--mode"}, &cMode,
      "This should be either 'limb-darkening', 'circles', 'linear', or 'smoothstep' (default: " +
          cMode + ").");
  args.addArgument({"--with-umbra"}, &settings.includeUmbra,
      "Add the umbra region to the shadow map (default: " + std::to_string(settings.includeUmbra) +
          ").");
  args.addArgument({"--mapping-exponent"}, &settings.mappingExponent,
      "Adjusts the distribution of sampling positions. A value of 1.0 will position the "
      "umbra's end in the middle of the texture, larger values will shift this to the "
      "right. (default: " +
          std::to_string(settings.mappingExponent) + ").");
  args.addArgument({"-h", "--help"}, &cPrintHelp, "Show this help message.");

  // Then do the actual parsing.
  try {
    std::vector<std::string> arguments(argv + 1, argv + argc);
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

  // Check whether a valid mode was given.
  if (cMode != "limb-darkening" && cMode != "circles" && cMode != "linear" &&
      cMode != "smoothstep") {
    std::cerr << "Invalid value given for --mode!" << std::endl;

    return 1;
  }

  // Initialize the limb darkening model.
  LimbDarkening limbDarkening;
  limbDarkening.init();

  // Initialize the global Cuda symbols.
  cudaMemcpyToSymbol(cShadowSettings, &settings, sizeof(ShadowSettings));
  cudaMemcpyToSymbol(cLimbDarkening, &limbDarkening, sizeof(LimbDarkening));

  // Compute the 2D kernel size.
  dim3     blockSize(16, 16);
  uint32_t numBlocksX = (settings.size + blockSize.x - 1) / blockSize.x;
  uint32_t numBlocksY = (settings.size + blockSize.y - 1) / blockSize.y;
  dim3     gridSize   = dim3(numBlocksX, numBlocksY);

  // Allocate the shared memory for the shadow map.
  float* shadow = nullptr;
  gpuErrchk(cudaMallocManaged(
      &shadow, static_cast<size_t>(settings.size * settings.size) * sizeof(float)));

  // Compute the shadow map based on the given mode.
  if (cMode == "limb-darkening") {
    computeLimbDarkeningShadow<<<gridSize, blockSize>>>(shadow);
  } else if (cMode == "circles") {
    computeCircleIntersectionShadow<<<gridSize, blockSize>>>(shadow);
  } else if (cMode == "linear") {
    computeLinearShadow<<<gridSize, blockSize>>>(shadow);
  } else if (cMode == "smoothstep") {
    computeSmoothstepShadow<<<gridSize, blockSize>>>(shadow);
  }

  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());

  // Finally write the output texture!
  stbi_write_hdr(
      cOutput.c_str(), static_cast<int>(settings.size), static_cast<int>(settings.size), 1, shadow);

  return 0;
}
