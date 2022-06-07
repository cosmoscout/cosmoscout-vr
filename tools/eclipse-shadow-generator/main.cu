////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2022 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "../../src/cs-utils/CommandLine.hpp"

#include "LimbDarkening.cuh"
#include "cudaErrorCheck.hpp"
#include "math.cuh"

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image.h>
#include <stb_image_write.h>

struct ShadowSettings {
  uint32_t size            = 512;
  bool     includeUmbra    = false;
  double   mappingExponent = 1.0;
};

__constant__ LimbDarkening  cLimbDarkening;
__constant__ ShadowSettings cShadowSettings;

__global__ void computeLimbDarkeningShadow(float* shadowMap) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int i = y * cShadowSettings.size + x;

  if ((x >= cShadowSettings.size) || (y >= cShadowSettings.size)) {
    return;
  }

  auto angles = math::mapPixelToAngles(glm::ivec2(x, y), cShadowSettings.size,
      cShadowSettings.mappingExponent, cShadowSettings.includeUmbra);

  double sunArea = math::getCircleArea(1.0);

  shadowMap[i] =
      1 - math::sampleCircleIntersection(1.0, angles.x, angles.y, cLimbDarkening) / sunArea;
}

__global__ void computeCircleIntersectionShadow(float* shadowMap) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int i = y * cShadowSettings.size + x;

  if ((x >= cShadowSettings.size) || (y >= cShadowSettings.size)) {
    return;
  }

  auto angles = math::mapPixelToAngles(glm::ivec2(x, y), cShadowSettings.size,
      cShadowSettings.mappingExponent, cShadowSettings.includeUmbra);

  double sunArea = math::getCircleArea(1.0);

  shadowMap[i] = 1 - math::getCircleIntersection(1.0, angles.x, angles.y) / sunArea;
}

__global__ void computeLinearShadow(float* shadowMap) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int i = y * cShadowSettings.size + x;

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

  shadowMap[i] = 1.0 - maxDepth * glm::clamp(1.0 - visiblePortion, 0.0, 1.0);
}

__global__ void computeSmoothstepShadow(float* shadowMap) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int i = y * cShadowSettings.size + x;

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

  shadowMap[i] =
      1.0 - maxDepth * glm::clamp(1.0 - glm::smoothstep(0.0, 1.0, visiblePortion), 0.0, 1.0);
}

// ---------------------------------------------------------------------------------------

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
    std::vector<std::string> arguments(argv + 2, argv + argc);
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

  if (cMode != "limb-darkening" && cMode != "circles" && cMode != "linear" &&
      cMode != "smoothstep") {
    std::cerr << "Invalid value given for --mode!" << std::endl;

    return 1;
  }

  LimbDarkening limbDarkening;
  limbDarkening.init();

  cudaMemcpyToSymbol(cShadowSettings, &settings, sizeof(ShadowSettings));
  cudaMemcpyToSymbol(cLimbDarkening, &limbDarkening, sizeof(LimbDarkening));

  uint32_t pixelCount = settings.size * settings.size;

  dim3 blockSize(16, 16);
  int  numBlocksX = (settings.size + blockSize.x - 1) / blockSize.x;
  int  numBlocksY = (settings.size + blockSize.y - 1) / blockSize.y;
  dim3 gridSize   = dim3(numBlocksX, numBlocksY);

  float* shadow;
  gpuErrchk(cudaMallocManaged(&shadow, pixelCount * sizeof(float)));

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

  stbi_write_hdr(
      cOutput.c_str(), static_cast<int>(settings.size), static_cast<int>(settings.size), 1, shadow);

  return 0;
}