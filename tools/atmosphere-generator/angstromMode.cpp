////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "angstromMode.hpp"

#include "common.hpp"

#include <glm/gtc/constants.hpp>
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

#include <fstream>

////////////////////////////////////////////////////////////////////////////////////////////////////

int angstromMode(std::vector<std::string> const& arguments) {

  bool        cPrintHelp              = false;
  std::string cOutput                 = "angstrom";
  double      cAlpha                  = 0.8;
  double      cBeta                   = 0.1;
  double      cSingleScatteringAlbedo = 0.7;
  double      cScaleHeight            = 1200.0;
  std::string cLambdas                = "";
  double      cMinLambda              = 0.36e-6;
  double      cMaxLambda              = 0.83e-6;
  int32_t     cLambdaSamples          = 15;

  // First configure all possible command line options.
  cs::utils::CommandLine args("Here are the available options:");
  args.addArgument({"-o", "--output"}, &cOutput,
      "The scattering data will be written to <name>_scattering.csv and "
      "<name>_absorption.csv, respectively (default: \"" +
          cOutput + "\").");
  args.addArgument({"--alpha"}, &cAlpha, fmt::format("The alpha parameter (default: {}).", cAlpha));
  args.addArgument({"--beta"}, &cBeta, fmt::format("The beta parameter (default: {}).", cBeta));
  args.addArgument({"--single-scattering-albedo"}, &cSingleScatteringAlbedo,
      fmt::format("The single-scattering albedo (default: {}).", cSingleScatteringAlbedo));
  args.addArgument({"--scale-height"}, &cScaleHeight,
      fmt::format("The scale height of the particles (default: {}).", cScaleHeight));
  common::addLambdaFlags(args, &cLambdas, &cMinLambda, &cMaxLambda, &cLambdaSamples);
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

  // Open the output files and write the CSV headers.
  std::ofstream scatteringOutput(cOutput + "_scattering.csv");
  std::ofstream absorptionOutput(cOutput + "_absorption.csv");

  scatteringOutput << "lambda,beta_sca" << std::endl;
  absorptionOutput << "lambda,beta_abs" << std::endl;

  // Now write a line to the CSV file for each wavelength.
  for (double lambda : lambdas) {

    double beta_ext = cBeta / std::pow(lambda * 1e6, cAlpha) / cScaleHeight;
    double beta_sca = cSingleScatteringAlbedo * beta_ext;
    double beta_abs = beta_ext - beta_sca;

    scatteringOutput << fmt::format("{},{}", lambda, beta_sca) << std::endl;
    absorptionOutput << fmt::format("{},{}", lambda, beta_abs) << std::endl;
  }

  return 0;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
