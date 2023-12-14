////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "rayleighMode.hpp"

#include "common.hpp"

#include <glm/gtc/constants.hpp>
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

#include <fstream>

////////////////////////////////////////////////////////////////////////////////////////////////////

int rayleighMode(std::vector<std::string> const& arguments) {

  bool        cPrintHelp      = false;
  std::string cOutput         = "rayleigh";
  bool        cPenndorf       = false;
  double      cIoR            = 1.00028276;
  double      cNumberDensity  = 2.68731e25;
  double      cDepolarization = 0.0;
  std::string cLambdas        = "";
  double      cMinLambda      = 0.36e-6;
  double      cMaxLambda      = 0.83e-6;
  int32_t     cLambdaSamples  = 15;
  int32_t     cThetaSamples   = 91;

  // First configure all possible command line options.
  cs::utils::CommandLine args("Here are the available options:");
  args.addArgument({"-o", "--output"}, &cOutput,
      "The scattering data will be written to <name>_phase.csv, <name>_scattering.csv, and "
      "<name>_absorption.csv, respectively (default: \"" +
          cOutput + "\").");
  args.addArgument({"--penndorf"}, &cPenndorf, "Use the Penndorf phase function (default: false)");
  args.addArgument(
      {"--ior"}, &cIoR, fmt::format("The index of refraction of the gas (default: {}).", cIoR));
  args.addArgument({"-n", "--number-density"}, &cNumberDensity,
      fmt::format("The number density per m³ (default: {}).", cNumberDensity));
  args.addArgument({"--depolarization"}, &cDepolarization,
      fmt::format(
          "The depolarization factor for the king-correction (default: {}).", cDepolarization));
  common::addLambdaFlags(args, &cLambdas, &cMinLambda, &cMaxLambda, &cLambdaSamples);
  common::addThetaFlags(args, &cThetaSamples);
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

  // Now assemble a list of wavelengths in m. This is either provided with the --lambda-samples
  // command-line parameter or via the combination of --min-lambda, --max-lambda, and
  // --lambda-samples.
  std::vector<double> lambdas =
      common::computeLambdas(cLambdas, cMinLambda, cMaxLambda, cLambdaSamples);

  if (lambdas.empty()) {
    return 1;
  }

  // We will write this many phase function samples. cThetaSamples determines the number of samples
  // between 0° and 90° (including both).
  int32_t totalAngles = cThetaSamples * 2 - 1;

  // Open the output files and write the CSV headers.
  std::ofstream phaseOutput(cOutput + "_phase.csv");
  std::ofstream scatteringOutput(cOutput + "_scattering.csv");
  std::ofstream absorptionOutput(cOutput + "_absorption.csv");

  scatteringOutput << "lambda,beta_sca" << std::endl;
  absorptionOutput << "lambda,beta_abs" << std::endl;
  phaseOutput << "lambda";
  for (int32_t t(0); t < totalAngles; ++t) {
    phaseOutput << fmt::format(",{}", 180.0 * t / (totalAngles - 1.0));
  }
  phaseOutput << std::endl;

  // Now write a line to the CSV file for each wavelength.
  for (double lambda : lambdas) {

    // Print scattering coefficient.
    double f = std::pow((cIoR * cIoR - 1.0), 2.0) * (6.0 + 3.0 * cDepolarization) /
               (6.0 - 7.0 * cDepolarization);
    double beta_sca =
        8.0 / 3.0 * std::pow(glm::pi<double>(), 3.0) * f / (cNumberDensity * std::pow(lambda, 4));

    scatteringOutput << fmt::format("{},{}", lambda, beta_sca) << std::endl;

    // Absorption is always zero.
    absorptionOutput << fmt::format("{},{}", lambda, 0.0) << std::endl;

    phaseOutput << fmt::format("{}", lambda);
    for (int32_t i(0); i < totalAngles; ++i) {
      double theta = i * glm::pi<double>() / (totalAngles - 1);
      double phase =
          cPenndorf
              ? 0.7629 / (4.0 * glm::pi<double>()) * (1.0 + 0.932 * std::pow(std::cos(theta), 2.0))
              : 3.0 / (16.0 * glm::pi<double>()) * (1.0 + std::pow(std::cos(theta), 2.0));
      phaseOutput << fmt::format(",{}", phase);
    }
    phaseOutput << std::endl;
  }

  return 0;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
