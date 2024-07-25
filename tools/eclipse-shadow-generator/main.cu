////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "../../src/cs-utils/CommandLine.hpp"

#include "advanced_modes.cuh"
#include "gpuErrCheck.hpp"
#include "simple_modes.cuh"

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image.h>
#include <stb_image_write.h>

// -------------------------------------------------------------------------------------------------

// clang-format off
void printHelp() {
  std::cout << "Welcome to the eclipse shadow texture generator! Usage:" << std::endl;
  std::cout << std::endl;
  std::cout << "  ./eclipse-shadow-generator <mode> <options>" << std::endl;
  std::cout << std::endl;
  std::cout << "There are different operation modes available. " << std::endl;
  std::cout << "Type './eclipse-shadow-generator <mode> --help' to learn more about a specific mode." << std::endl;
  std::cout << std::endl;
  std::cout << "These modes are available:" << std::endl;
  std::cout << "bruneton       Computes an eclipse shadow with atmospheric effects based on the Bruneton atmospheric model." << std::endl;
  std::cout << "limp-darkening Computes an eclipse shadow without atmospheric effects using sampling of a limb-darkening function." << std::endl;
  std::cout << "circles        Computes an eclipse shadow without atmospheric effects based on circle intersections." << std::endl;
  std::cout << "smoothstep     Computes an eclipse shadow without atmospheric effects using a smoothstep function in the penumbra on circle intersections." << std::endl;
  std::cout << "linear         Computes an eclipse shadow without atmospheric effects with a linear brightness gradient." << std::endl;
  std::cout << "planet-view    Computes a view of a planet as seen from space." << std::endl;
  std::cout << "atmo-view      Computes a view of the entire atmosphere from a given position in space." << std::endl;
}
// clang-format on

////////////////////////////////////////////////////////////////////////////////////////////////////
// This tool can be used to create the eclipse shadow maps used by CosmoScout VR. See the         //
// README.md file in this directory for usage instructions!                                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char** argv) {

  stbi_flip_vertically_on_write(1);

  if (argc <= 1) {
    printHelp();
    return 0;
  }

  std::string cMode(argv[1]);

  std::vector<std::string> arguments(argv + 2, argv + argc);

  if (cMode == "limb-darkening") {
    return simple::limbDarkeningMode(arguments);
  }

  if (cMode == "circles") {
    return simple::circleIntersectionMode(arguments);
  }

  if (cMode == "smoothstep") {
    return simple::smoothstepMode(arguments);
  }

  if (cMode == "linear") {
    return simple::linearMode(arguments);
  }

  if (cMode == "bruneton") {
    return advanced::brunetonMode(arguments);
  }

  if (cMode == "planet-view") {
    return advanced::planetViewMode(arguments);
  }

  if (cMode == "atmo-view") {
    return advanced::atmoViewMode(arguments);
  }

  printHelp();

  return 0;
}
