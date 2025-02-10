////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "ozoneMode.hpp"

#include "../../../src/cs-utils/utils.hpp"
#include "common.hpp"

#include <glm/gtc/constants.hpp>
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

#include <fstream>

////////////////////////////////////////////////////////////////////////////////////////////////////

int ozoneMode(std::vector<std::string> const& arguments) {

  bool        cPrintHelp     = false;
  std::string cOutput        = "ozone";
  double      cNumberDensity = 4e18; // See https://amt.copernicus.org/articles/14/6057/2021/
  std::string cLambdas       = "";
  double      cMinLambda     = 0.36e-6;
  double      cMaxLambda     = 0.83e-6;
  int32_t     cLambdaSamples = 15;

  // First configure all possible command line options.
  cs::utils::CommandLine args("Lambdas are only valid between 0.36 "
                              "and 0.83. The Here are the available options:");
  args.addArgument({"-o", "--output"}, &cOutput,
      "The absorption data will be written to <name>_absorption.csv (default: \"" + cOutput +
          "\").");
  args.addArgument({"--number-density"}, &cNumberDensity,
      fmt::format("The peak number of particles per m³ (default: {}).", cNumberDensity));
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

  // Values from
  // http://www.iup.uni-bremen.de/gruppen/molspec/databases/referencespectra/o3spectra2011/index.html
  // for 233K, summed and averaged in each bin (e.g. the value for 360nm is the average of the
  // original values for all wavelengths between 360 and 370nm). Values in m². The data has been
  // updated as newer data became available on the website.
  const double              minLambda   = 0.36e-6;
  const double              maxLambda   = 0.83e-6;
  const std::vector<double> absorptions = {1.18006e-27, 2.18205e-28, 2.81764e-28, 6.63629e-28,
      1.52685e-27, 2.76259e-27, 5.51975e-27, 8.45102e-27, 1.58232e-26, 2.31555e-26, 3.66625e-26,
      4.92413e-26, 7.76088e-26, 9.02900e-26, 1.48333e-25, 1.60547e-25, 2.14349e-25, 2.76161e-25,
      3.09823e-25, 3.50934e-25, 4.27703e-25, 4.68477e-25, 4.40965e-25, 4.71385e-25, 5.03275e-25,
      4.31623e-25, 3.74999e-25, 3.22324e-25, 2.66819e-25, 2.24367e-25, 1.85651e-25, 1.47571e-25,
      1.21105e-25, 9.43730e-26, 7.46292e-26, 6.57117e-26, 5.10619e-26, 4.14823e-26, 4.22622e-26,
      3.23257e-26, 2.44425e-26, 2.79549e-26, 2.52744e-26, 1.61447e-26, 1.45506e-26, 2.07028e-26,
      1.37295e-26, 6.98672e-27};

  // Open the output file and write the CSV header.
  std::ofstream output(cOutput + "_absorption.csv");
  output << "lambda,beta_abs" << std::endl;

  // Now that we have a list of wavelengths, compute the absorption cross-sections at each
  // wavelength using linear interpolation.
  for (double lambda : lambdas) {
    double absorption = common::interpolate(absorptions, minLambda, maxLambda, lambda);
    output << fmt::format("{},{}", lambda, absorption * cNumberDensity) << std::endl;
  }

  return 0;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
