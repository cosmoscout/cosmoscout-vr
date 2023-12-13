////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "../../src/cs-utils/CommandLine.hpp"

#include "densityMode.hpp"
#include "mieMode.hpp"
#include "ozoneMode.hpp"
#include "rayleighMode.hpp"

// -------------------------------------------------------------------------------------------------

// clang-format off
void printHelp() {
  std::cout << "Welcome to the atmosphere generator! Usage:" << std::endl;
  std::cout << std::endl;
  std::cout << "  ./atmosphere-generator <mode> <options>" << std::endl;
  std::cout << std::endl;
  std::cout << "There are different operation modes available. " << std::endl;
  std::cout << "Type './atmosphere-generator <mode> --help' to learn more about a specific mode." << std::endl;
  std::cout << std::endl;
  std::cout << "These modes are available:" << std::endl;
  std::cout << "mie       Precompute phase functions and scattering cross-sections using Mie theory." << std::endl;
  std::cout << "rayleigh  Write Rayleigh phase function and scattering cross-sections for the given wavelengths." << std::endl;
  std::cout << "ozone     Write ozone absorption cross-sections for the given wavelengths." << std::endl;
  std::cout << "density   Precompute particle density distributions as a function of altitude." << std::endl;
}
// clang-format on

////////////////////////////////////////////////////////////////////////////////////////////////////
// This tool can be used to create the atmospheric data used by CosmoScout VR. See the            //
// README.md file in this directory for usage instructions!                                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char** argv) {

  if (argc <= 1) {
    printHelp();
    return 0;
  }

  std::string cMode(argv[1]);

  std::vector<std::string> arguments(argv + 2, argv + argc);

  if (cMode == "mie") {
    return mieMode(arguments);
  }

  if (cMode == "rayleigh") {
    return rayleighMode(arguments);
  }

  if (cMode == "ozone") {
    return ozoneMode(arguments);
  }

  if (cMode == "density") {
    return densityMode(arguments);
  }

  printHelp();

  return 0;
}
