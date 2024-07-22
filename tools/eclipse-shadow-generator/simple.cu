////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "LimbDarkening.cuh"
#include "common.hpp"
#include "gpuErrCheck.hpp"
#include "math.cuh"
#include "simple.cuh"

#include <stb_image_write.h>

#include <cstdint>

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace {

////////////////////////////////////////////////////////////////////////////////////////////////////

enum class Mode { eLimbDarkening, eCircleIntersection, eLinear, eSmoothstep };

////////////////////////////////////////////////////////////////////////////////////////////////////

__device__ void writeAsRGBValue(float value, float* shadowMap, uint32_t index) {
  shadowMap[index * 3 + 0] = value;
  shadowMap[index * 3 + 1] = value;
  shadowMap[index * 3 + 2] = value;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void computeLimbDarkeningShadow(float* shadowMap, common::ShadowSettings shadow,
    common::OutputSettings output, LimbDarkening limbDarkening) {
  uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
  uint32_t i = y * output.size + x;

  if ((x >= output.size) || (y >= output.size)) {
    return;
  }

  auto angles = math::mapPixelToAngles(
      glm::ivec2(x, y), output.size, shadow.mappingExponent, shadow.includeUmbra);

  double sunArea = math::getCircleArea(1.0);

  float intensity = static_cast<float>(
      1 - math::sampleCircleIntersection(1.0, angles.x, angles.y, limbDarkening) / sunArea);

  writeAsRGBValue(intensity, shadowMap, i);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void computeCircleIntersectionShadow(
    float* shadowMap, common::ShadowSettings shadow, common::OutputSettings output) {
  uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
  uint32_t i = y * output.size + x;

  if ((x >= output.size) || (y >= output.size)) {
    return;
  }

  auto angles = math::mapPixelToAngles(
      glm::ivec2(x, y), output.size, shadow.mappingExponent, shadow.includeUmbra);

  double sunArea = math::getCircleArea(1.0);

  float intensity =
      static_cast<float>(1.0 - math::getCircleIntersection(1.0, angles.x, angles.y) / sunArea);

  writeAsRGBValue(intensity, shadowMap, i);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void computeLinearShadow(
    float* shadowMap, common::ShadowSettings shadow, common::OutputSettings output) {
  uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
  uint32_t i = y * output.size + x;

  if ((x >= output.size) || (y >= output.size)) {
    return;
  }

  auto angles = math::mapPixelToAngles(
      glm::ivec2(x, y), output.size, shadow.mappingExponent, shadow.includeUmbra);

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

__global__ void computeSmoothstepShadow(
    float* shadowMap, common::ShadowSettings shadow, common::OutputSettings output) {
  uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
  uint32_t i = y * output.size + x;

  if ((x >= output.size) || (y >= output.size)) {
    return;
  }

  auto angles = math::mapPixelToAngles(
      glm::ivec2(x, y), output.size, shadow.mappingExponent, shadow.includeUmbra);

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

int run(Mode mode, std::vector<std::string> const& arguments) {
  common::ShadowSettings shadow;
  common::OutputSettings output;
  bool                   cPrintHelp = false;

  // First configure all possible command line options.
  cs::utils::CommandLine args("Here are the available options:");
  common::addShadowSettingsFlags(args, shadow);
  common::addOutputSettingsFlags(args, output);
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

  // Initialize the limb darkening model.
  LimbDarkening limbDarkening;
  limbDarkening.init();

  // Compute the 2D kernel size.
  dim3     blockSize(16, 16);
  uint32_t numBlocksX = (output.size + blockSize.x - 1) / blockSize.x;
  uint32_t numBlocksY = (output.size + blockSize.y - 1) / blockSize.y;
  dim3     gridSize   = dim3(numBlocksX, numBlocksY);

  // Allocate the shared memory for the shadow map.
  float* shadowMap = nullptr;
  gpuErrchk(cudaMallocManaged(
      &shadowMap, static_cast<size_t>(output.size * output.size) * 3 * sizeof(float)));

  if (mode == Mode::eLimbDarkening) {
    computeLimbDarkeningShadow<<<gridSize, blockSize>>>(shadowMap, shadow, output, limbDarkening);
  } else if (mode == Mode::eCircleIntersection) {
    computeCircleIntersectionShadow<<<gridSize, blockSize>>>(shadowMap, shadow, output);
  } else if (mode == Mode::eLinear) {
    computeLinearShadow<<<gridSize, blockSize>>>(shadowMap, shadow, output);
  } else if (mode == Mode::eSmoothstep) {
    computeSmoothstepShadow<<<gridSize, blockSize>>>(shadowMap, shadow, output);
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

namespace simple {

////////////////////////////////////////////////////////////////////////////////////////////////////

int limbDarkeningMode(std::vector<std::string> const& arguments) {
  return run(Mode::eLimbDarkening, arguments);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

int circleIntersectionMode(std::vector<std::string> const& arguments) {
  return run(Mode::eCircleIntersection, arguments);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

int linearMode(std::vector<std::string> const& arguments) {
  return run(Mode::eLinear, arguments);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

int smoothstepMode(std::vector<std::string> const& arguments) {
  return run(Mode::eSmoothstep, arguments);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace simple