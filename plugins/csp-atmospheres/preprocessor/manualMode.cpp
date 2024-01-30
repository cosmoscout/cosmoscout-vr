////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "manualMode.hpp"

#include "common.hpp"

#include <glm/gtc/constants.hpp>
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

#include <fstream>

////////////////////////////////////////////////////////////////////////////////////////////////////

int manualMode(std::vector<std::string> const& arguments) {

  bool        cPrintHelp     = false;
  std::string cOutput        = "manual";
  std::string cQuantity      = "";
  std::string cValues        = "";
  std::string cLambdas       = "";
  double      cMinLambda     = 0.36e-6;
  double      cMaxLambda     = 0.83e-6;
  int32_t     cLambdaSamples = 15;

  // First configure all possible command line options.
  cs::utils::CommandLine args("Here are the available options:");
  args.addArgument({"-o", "--output"}, &cOutput,
      "The data will be written to <name>.csv (default: \"" + cOutput + "\").");
  args.addArgument({"--quantity"}, &cQuantity,
      "The header string in the output CSV file. Usually this should be either beta_sca or "
      "beta_abs.");
  args.addArgument({"--values"}, &cValues,
      "The numbers to write into the CSV file. Should be a comma-separated list of one value per "
      "wavelength.");
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

  // Some parameters are mandatory.
  if (cQuantity.empty() || cValues.empty()) {
    std::cerr
        << "Please specify a quantity name and the values with the --quantity and --values options!"
        << std::endl;
    return 1;
  }

  // Now assemble a list of wavelengths in m. This is either provided with the --lambda-samples
  // command-line parameter or via the combination of --min-lambda, --max-lambda, and
  // --lambda-samples.
  std::vector<double> lambdas =
      common::computeLambdas(cLambdas, cMinLambda, cMaxLambda, cLambdaSamples);

  if (lambdas.empty()) {
    return 1;
  }

  // Now parse the provided list of values.
  std::vector<double> values = common::parseNumberList(cValues);

  if (values.size() != 1 && values.size() != lambdas.size()) {
    std::cerr << "There must be either one value parameter or one per wavelength!" << std::endl;
    return 1;
  }

  // Open the output file and write the CSV header.
  std::ofstream output(cOutput + ".csv");

  output << "lambda," << cQuantity << std::endl;

  // Now write a line to the CSV file for each wavelength.
  for (size_t i(0); i < lambdas.size(); ++i) {
    double lambda = lambdas[i];
    double value  = values.size() == 1 ? values[0] : values[i];

    output << fmt::format("{},{}", lambda, value) << std::endl;
  }

  return 0;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
