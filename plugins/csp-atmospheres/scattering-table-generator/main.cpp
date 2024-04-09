////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "../../../src/cs-utils/CommandLine.hpp"

#include "angstromMode.hpp"
#include "densityMode.hpp"
#include "hulstMode.hpp"
#include "manualMode.hpp"
#include "mieMode.hpp"
#include "ozoneMode.hpp"
#include "parametricModes.hpp"
#include "rayleighMode.hpp"

// -------------------------------------------------------------------------------------------------

// clang-format off
void printHelp() {
  std::cout << "Welcome to the CSV Generator! Usage:" << std::endl;
  std::cout << std::endl;
  std::cout << "  ./scattering-table-generator <mode> <options>" << std::endl;
  std::cout << std::endl;
  std::cout << "There are different operation modes available. " << std::endl;
  std::cout << "Type './scattering-table-generator <mode> --help' to learn more about a specific mode." << std::endl;
  std::cout << std::endl;
  std::cout << "These modes are available:" << std::endl;
  std::cout << "mie       Precompute phase functions and scattering coefficients using Mie theory." << std::endl;
  std::cout << "rayleigh  Write the Rayleigh phase function and scattering coefficients for the given wavelengths." << std::endl;
  std::cout << "angstrom  Write the scattering and absorption coefficients using Ångström's turbidity formula." << std::endl;
  std::cout << "hulst     Write the scattering and absorption coefficients using approximations by Van de Hulst." << std::endl;
  std::cout << "manual    Write any scattering and absorption coefficients for the given wavelengths." << std::endl;
  std::cout << "cornette  Write the Cornette-Shanks phase function for the given wavelengths." << std::endl;
  std::cout << "henyey    Write the Henyey-Greenstein phase function for the given wavelengths." << std::endl;
  std::cout << "dhenyey   Write the Double-Henyey-Greenstein phase function for the given wavelengths." << std::endl;
  std::cout << "ozone     Write ozone absorption coefficients for the given wavelengths." << std::endl;
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

  if (cMode == "angstrom") {
    return angstromMode(arguments);
  }

  if (cMode == "hulst") {
    return hulstMode(arguments);
  }

  if (cMode == "manual") {
    return manualMode(arguments);
  }

  if (cMode == "cornette") {
    return cornetteShanksMode(arguments);
  }

  if (cMode == "henyey") {
    return henyeyGreensteinMode(arguments);
  }

  if (cMode == "dhenyey") {
    return doubleHenyeyGreensteinMode(arguments);
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
