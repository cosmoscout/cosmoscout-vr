////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "common.hpp"

#include "../../src/cs-utils/utils.hpp"

#include <spdlog/spdlog.h>

namespace common {

////////////////////////////////////////////////////////////////////////////////////////////////////

void addLambdaFlags(cs::utils::CommandLine& commandLine, std::string* lambdas, double* minLambda,
    double* maxLambda, int32_t* lambdaSamples) {

  commandLine.addArgument({"--min-lambda"}, minLambda,
      fmt::format("The minimum wavelength in m (default: {})", *minLambda));
  commandLine.addArgument({"--max-lambda"}, maxLambda,
      fmt::format("The maximum wavelength in m (default: {})", *maxLambda));
  commandLine.addArgument({"--lambda-samples"}, lambdaSamples,
      fmt::format("The number of wavelengths to compute (default: {})", *lambdaSamples));
  commandLine.addArgument({"--lambdas"}, lambdas,
      "A comma-separated list of wavelengths in m. If provided, --min-lambda, --max-lambda, and "
      "--lambda-samples are ignored.");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void addThetaFlags(cs::utils::CommandLine& commandLine, int32_t* thetaSamples) {
  commandLine.addArgument({"--theta-samples"}, thetaSamples,
      "The number of angles to compute between 0° and 90° (default: " +
          std::to_string(*thetaSamples) + ").");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::vector<double> computeLambdas(
    std::string const& lambdas, double minLambda, double maxLambda, int32_t lambdaSamples) {
  std::vector<double> result;

  if (lambdas.empty()) {
    if (lambdaSamples <= 0) {
      std::cerr << "Lambda-sample count must be > 0!" << std::endl;
      return result;
    } else if (lambdaSamples == 1) {
      result.push_back(minLambda);
    } else {
      for (int32_t i(0); i < lambdaSamples; ++i) {
        result.push_back(minLambda + (maxLambda - minLambda) * i / (lambdaSamples - 1.0));
      }
    }
  } else {
    auto tokens = cs::utils::splitString(lambdas, ',');
    for (auto token : tokens) {
      result.push_back(cs::utils::fromString<double>(token));
    }
  }

  return result;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::vector<double> parseNumberList(std::string const& list) {
  std::vector<double> result;

  auto tokens = cs::utils::splitString(list, ',');
  for (auto token : tokens) {
    result.push_back(cs::utils::fromString<double>(token));
  }

  return result;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace common
