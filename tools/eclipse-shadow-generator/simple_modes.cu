////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "simple_modes.cuh"

#include "LimbDarkening.cuh"
#include "common.hpp"
#include "gpuErrCheck.hpp"
#include "math.cuh"
#include "tiff_utils.hpp"

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace {

////////////////////////////////////////////////////////////////////////////////////////////////////

enum class Mode { eLimbDarkening, eCircleIntersection, eLinear, eSmoothstep };

////////////////////////////////////////////////////////////////////////////////////////////////////

__device__ void writeAsRGBValue(float value, float* buffer, uint32_t index) {
  buffer[index * 3 + 0] = value;
  buffer[index * 3 + 1] = value;
  buffer[index * 3 + 2] = value;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void computeLimbDarkeningShadow(
    common::Mapping mapping, common::Output output, common::LimbDarkening limbDarkening) {
  uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
  uint32_t i = y * output.mSize + x;

  if ((x >= output.mSize) || (y >= output.mSize)) {
    return;
  }

  double radiusOcc, distance;
  math::mapPixelToRadii(glm::ivec2(x, y), output.mSize, mapping, radiusOcc, distance);

  double sunArea   = math::getCircleArea(1.0);
  float  intensity = static_cast<float>(
      1 - math::sampleCircleIntersection(1.0, radiusOcc, distance, limbDarkening) / sunArea);

  writeAsRGBValue(intensity, output.mBuffer, i);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void computeCircleIntersectionShadow(common::Mapping mapping, common::Output output) {
  uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
  uint32_t i = y * output.mSize + x;

  if ((x >= output.mSize) || (y >= output.mSize)) {
    return;
  }

  double radiusOcc, distance;
  math::mapPixelToRadii(glm::ivec2(x, y), output.mSize, mapping, radiusOcc, distance);

  double sunArea = math::getCircleArea(1.0);
  float  intensity =
      static_cast<float>(1.0 - math::getCircleIntersection(1.0, radiusOcc, distance) / sunArea);

  writeAsRGBValue(intensity, output.mBuffer, i);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void computeLinearShadow(common::Mapping mapping, common::Output output) {
  uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
  uint32_t i = y * output.mSize + x;

  if ((x >= output.mSize) || (y >= output.mSize)) {
    return;
  }

  double radiusSun = 1.0;
  double radiusOcc, distance;
  math::mapPixelToRadii(glm::ivec2(x, y), output.mSize, mapping, radiusOcc, distance);

  double visiblePortion = (distance - glm::abs(radiusSun - radiusOcc)) /
                          (radiusSun + radiusOcc - glm::abs(radiusSun - radiusOcc));

  double maxDepth = glm::min(1.0, glm::pow(radiusOcc / radiusSun, 2.0));

  float intensity = static_cast<float>(1.0 - maxDepth * glm::clamp(1.0 - visiblePortion, 0.0, 1.0));

  writeAsRGBValue(intensity, output.mBuffer, i);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void computeSmoothstepShadow(common::Mapping mapping, common::Output output) {
  uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
  uint32_t i = y * output.mSize + x;

  if ((x >= output.mSize) || (y >= output.mSize)) {
    return;
  }

  double radiusSun = 1.0;
  double radiusOcc, distance;
  math::mapPixelToRadii(glm::ivec2(x, y), output.mSize, mapping, radiusOcc, distance);

  double visiblePortion = (distance - glm::abs(radiusSun - radiusOcc)) /
                          (radiusSun + radiusOcc - glm::abs(radiusSun - radiusOcc));

  double maxDepth = glm::min(1.0, glm::pow(radiusOcc / radiusSun, 2.0));

  float intensity = static_cast<float>(
      1.0 - maxDepth * glm::clamp(1.0 - glm::smoothstep(0.0, 1.0, visiblePortion), 0.0, 1.0));

  writeAsRGBValue(intensity, output.mBuffer, i);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

int run(Mode mode, std::vector<std::string> const& arguments) {
  common::Mapping mapping;
  common::Output  output;
  bool            cPrintHelp = false;

  // First configure all possible command line options.
  cs::utils::CommandLine args("Here are the available options:");
  common::addMappingFlags(args, mapping);
  common::addOutputFlags(args, output);
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

  if (mode == Mode::eLimbDarkening) {
    computeLimbDarkeningShadow<<<gridSize, blockSize>>>(mapping, output, limbDarkening);
  } else if (mode == Mode::eCircleIntersection) {
    computeCircleIntersectionShadow<<<gridSize, blockSize>>>(mapping, output);
  } else if (mode == Mode::eLinear) {
    computeLinearShadow<<<gridSize, blockSize>>>(mapping, output);
  } else if (mode == Mode::eSmoothstep) {
    computeSmoothstepShadow<<<gridSize, blockSize>>>(mapping, output);
  }

  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());

  // Finally write the output texture!
  tiff_utils::write2D(output.mFile, output.mBuffer, static_cast<int>(output.mSize),
      static_cast<int>(output.mSize), 3);

  // Free the shared memory.
  gpuErrchk(cudaFree(output.mBuffer));

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