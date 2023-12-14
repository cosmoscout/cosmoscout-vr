////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "parametricModes.hpp"

#include "common.hpp"

#include "../../src/cs-utils/utils.hpp"

#include <glm/gtc/constants.hpp>
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

#include <fstream>

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace {

////////////////////////////////////////////////////////////////////////////////////////////////////

enum class Type {
  eCornetteShanks,
  eHenyeyGreenstein,
  eDoubleHenyeyGreenstein,
};

int impl(std::vector<std::string> const& arguments, Type type) {

  const std::unordered_map<Type, std::string> defaultOutputs = {
      {Type::eCornetteShanks, "cornette_shanks"}, {Type::eHenyeyGreenstein, "henyey_greenstein"},
      {Type::eDoubleHenyeyGreenstein, "double_henyey_greenstein"}};

  bool        cPrintHelp     = false;
  std::string cOutput        = defaultOutputs.at(type);
  std::string cG1            = "0.8";
  std::string cG2            = "0.1";
  std::string cAlpha         = "0.3";
  std::string cLambdas       = "";
  double      cMinLambda     = 0.36e-6;
  double      cMaxLambda     = 0.83e-6;
  int32_t     cLambdaSamples = 15;
  int32_t     cThetaSamples  = 91;

  // First configure all possible command line options.
  cs::utils::CommandLine args("Here are the available options:");
  args.addArgument({"-o", "--output"}, &cOutput,
      "The phase functions will be written to <name>_phase.csv (default: \"" + cOutput + "\").");

  if (type == Type::eDoubleHenyeyGreenstein) {
    args.addArgument({"--g1"}, &cG1,
        "The first g parameter. This can be a comma-separated list of values matching the number "
        "of "
        "wavelengths (default: " +
            cG1 + ").");
    args.addArgument({"--g2"}, &cG2,
        "The second g parameter. This can be a comma-separated list of values matching the number "
        "of "
        "wavelengths (default: " +
            cG2 + ").");
    args.addArgument({"--alpha"}, &cAlpha,
        "The alpha parameter. This can be a comma-separated list of values matching the number of "
        "wavelengths (default: " +
            cAlpha + ").");
  } else {
    args.addArgument({"--g"}, &cG1,
        "The g parameter. This can be a comma-separated list of values matching the number of "
        "wavelengths (default: " +
            cG1 + ").");
  }

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

  std::vector<double> g1s    = common::parseNumberList(cG1);
  std::vector<double> g2s    = common::parseNumberList(cG2);
  std::vector<double> alphas = common::parseNumberList(cAlpha);

  if (type == Type::eDoubleHenyeyGreenstein) {
    if (g1s.size() != 1 && g1s.size() != lambdas.size()) {
      std::cerr << "There must be either one g1 parameter or one per wavelength!" << std::endl;
      return 1;
    }
    if (g2s.size() != 1 && g2s.size() != lambdas.size()) {
      std::cerr << "There must be either one g2 parameter or one per wavelength!" << std::endl;
      return 1;
    }
    if (alphas.size() != 1 && alphas.size() != lambdas.size()) {
      std::cerr << "There must be either one alpha parameter or one per wavelength!" << std::endl;
      return 1;
    }
  } else {
    if (g1s.size() != 1 && g1s.size() != lambdas.size()) {
      std::cerr << "There must be either one g parameter or one per wavelength!" << std::endl;
      return 1;
    }
  }

  // We will write this many phase function samples. cThetaSamples determines the number of samples
  // between 0° and 90° (including both).
  int32_t totalAngles = cThetaSamples * 2 - 1;

  // Open the output files and write the CSV headers.
  std::ofstream output(cOutput + "_phase.csv");

  output << "lambda";
  for (int32_t t(0); t < totalAngles; ++t) {
    output << fmt::format(",{}", 180.0 * t / (totalAngles - 1.0));
  }
  output << std::endl;

  // Now write a line to the CSV file for each wavelength.
  for (size_t i(0); i < lambdas.size(); ++i) {
    double lambda = lambdas[i];
    double g1     = g1s.size() == 1 ? g1s[0] : g1s[i];
    double g2     = g2s.size() == 1 ? g2s[0] : g1s[i];
    double alpha  = alphas.size() == 1 ? alphas[0] : alphas[i];

    output << fmt::format("{}", lambda);
    for (int32_t i(0); i < totalAngles; ++i) {
      double theta = i * glm::pi<double>() / (totalAngles - 1);
      double phase = 0.0;

      if (type == Type::eCornetteShanks) {

        double mu = std::cos(theta);
        double a  = 3.0 * (1.0 - g1 * g1) * (1.0 + mu * mu);
        double b  = 8.0 * glm::pi<double>() * (2.0 + g1 * g1) *
                   std::pow(1.0 + g1 * g1 - 2.0 * g1 * mu, 1.5);
        phase = a / b;

      } else if (type == Type::eHenyeyGreenstein) {

        phase = (1.0 - g1 * g1) / std::pow(1.0 - 2.0 * g1 * std::cos(theta) + g1 * g1, 1.5) /
                (4.0 * glm::pi<double>());

      } else if (type == Type::eDoubleHenyeyGreenstein) {

        double a = (1.0 - g1 * g1) / std::pow(1.0 - 2.0 * g1 * std::cos(theta) + g1 * g1, 1.5) /
                   (4.0 * glm::pi<double>());
        double b = (1.0 - g2 * g2) / std::pow(1.0 - 2.0 * g2 * std::cos(theta) + g2 * g2, 1.5) /
                   (4.0 * glm::pi<double>());
        phase = alpha * a + (1.0 - alpha) * b;
      }

      output << fmt::format(",{}", phase);
    }
    output << std::endl;
  }

  return 0;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace

////////////////////////////////////////////////////////////////////////////////////////////////////

int cornetteShanksMode(std::vector<std::string> const& arguments) {
  return impl(arguments, Type::eCornetteShanks);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

int henyeyGreensteinMode(std::vector<std::string> const& arguments) {
  return impl(arguments, Type::eHenyeyGreenstein);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

int doubleHenyeyGreensteinMode(std::vector<std::string> const& arguments) {
  return impl(arguments, Type::eDoubleHenyeyGreenstein);
}

////////////////////////////////////////////////////////////////////////////////////////////////////