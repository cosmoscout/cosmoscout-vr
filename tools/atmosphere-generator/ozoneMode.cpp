////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "ozoneMode.hpp"

#include "../../src/cs-utils/utils.hpp"
#include "common.hpp"

#include <glm/gtc/constants.hpp>
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

#include <fstream>

////////////////////////////////////////////////////////////////////////////////////////////////////

int ozoneMode(std::vector<std::string> const& arguments) {

  bool        cPrintHelp     = false;
  std::string cOutput        = "ozone";
  std::string cLambdas       = "";
  double      cMinLambda     = 0.36;
  double      cMaxLambda     = 0.83;
  int32_t     cLambdaSamples = 15;

  // First configure all possible command line options.
  cs::utils::CommandLine args("Welcome to the ozone writer! Lambdas are only valid between 0.36 "
                              "and 0.83. The Here are the available options:");
  args.addArgument({"-o", "--output"}, &cOutput,
      "The absorption data will be written to <name>_absorption.csv (default: \"" + cOutput +
          "\").");
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

  // Now assemble a list of wavelengths in µm. This is either provided with the --lambda-samples
  // command-line parameter or via the combination of --min-lambda, --max-lambda, and
  // --lambda-samples.
  std::vector<double> lambdas =
      common::computeLambdas(cLambdas, cMinLambda, cMaxLambda, cLambdaSamples);

  if (lambdas.empty()) {
    return 1;
  }

  // Values from
  // http://www.iup.uni-bremen.de/gruppen/molspec/databases/referencespectra/o3spectra2011/index.html
  // for 233K, summed and averaged in each bin (e.g. the value for 360nm is the average of the
  // original values for all wavelengths between 360 and 370nm). Values in m².
  const double              minLambda   = 0.36;
  const double              maxLambda   = 0.83;
  const std::vector<double> absorptions = {1.18e-27, 2.182e-28, 2.818e-28, 6.636e-28, 1.527e-27,
      2.763e-27, 5.52e-27, 8.451e-27, 1.582e-26, 2.316e-26, 3.669e-26, 4.924e-26, 7.752e-26,
      9.016e-26, 1.48e-25, 1.602e-25, 2.139e-25, 2.755e-25, 3.091e-25, 3.5e-25, 4.266e-25,
      4.672e-25, 4.398e-25, 4.701e-25, 5.019e-25, 4.305e-25, 3.74e-25, 3.215e-25, 2.662e-25,
      2.238e-25, 1.852e-25, 1.473e-25, 1.209e-25, 9.423e-26, 7.455e-26, 6.566e-26, 5.105e-26,
      4.15e-26, 4.228e-26, 3.237e-26, 2.451e-26, 2.801e-26, 2.534e-26, 1.624e-26, 1.465e-26,
      2.078e-26, 1.383e-26, 7.105e-27};

  // Open the output file and write the CSV header.
  std::ofstream output(cOutput + "_absorption.csv");
  output << "lambda,c_abs" << std::endl;

  // Now that we have a list of wavelengths, compute the absorption cross-sections at each
  // wavelength using linear interpolation.
  for (double lambda : lambdas) {
    int32_t maxIndex = static_cast<int32_t>(absorptions.size());
    int32_t lowerIndex =
        static_cast<int32_t>((maxIndex - 1) * (lambda - minLambda) / (maxLambda - minLambda));
    lowerIndex = std::max(0, std::min(maxIndex - 1, lowerIndex));

    int32_t upperIndex = std::min(maxIndex - 1, lowerIndex + 1);

    double lowerLambda = minLambda + lowerIndex * (maxLambda - minLambda) / (maxIndex - 1);
    double upperLambda = minLambda + upperIndex * (maxLambda - minLambda) / (maxIndex - 1);

    double lowerAbsorption = absorptions[lowerIndex];
    double upperAbsorption = absorptions[upperIndex];

    double alpha =
        lowerIndex == upperIndex ? 0.0 : (lambda - lowerLambda) / (upperLambda - lowerLambda);
    alpha = std::max(0.0, std::min(1.0, alpha));

    double absorption = (1.0 - alpha) * lowerAbsorption + alpha * upperAbsorption;

    // We write data in µm².
    output << fmt::format("{},{}", lambda, absorption * 1e12) << std::endl;
  }

  return 0;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
