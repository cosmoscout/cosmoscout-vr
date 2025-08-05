////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "../../src/cs-utils/CommandLine.hpp"

#include "advanced_modes.cuh"
#include "gpuErrCheck.hpp"
#include "simple_modes.cuh"

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
  std::cout << "limb-luminance Computes the average luminance of atmosphere for each position in the shadow map in a direction-dependent manner." << std::endl;
}
// clang-format on

////////////////////////////////////////////////////////////////////////////////////////////////////
// This tool can be used to create the eclipse shadow maps used by CosmoScout VR. See the         //
// README.md file in this directory for usage instructions!                                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char** argv) {

  if (argc <= 1) {
    printHelp();
    return 0;
  }

  std::string cMode(argv[1]);

  std::vector<std::string> arguments(argv + 2, argv + argc);

  if (cMode == "simple-limb-darkening") {
    return simple::limbDarkeningMode(arguments);
  }

  if (cMode == "simple-circles") {
    return simple::circleIntersectionMode(arguments);
  }

  if (cMode == "simple-smoothstep") {
    return simple::smoothstepMode(arguments);
  }

  if (cMode == "simple-linear") {
    return simple::linearMode(arguments);
  }

  if (cMode == "advanced-shadow") {
    return advanced::shadowMode(arguments);
  }

  if (cMode == "advanced-limb-luminance") {
    return advanced::limbLuminanceMode(arguments);
  }

  if (cMode == "advanced-planet-view") {
    return advanced::planetViewMode(arguments);
  }

  if (cMode == "advanced-atmo-view") {
    return advanced::atmoViewMode(arguments);
  }

  printHelp();

  return 0;
}
