////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "common.hpp"

#include "../../src/cs-utils/utils.hpp"

namespace common {

////////////////////////////////////////////////////////////////////////////////////////////////////

void addLambdaFlags(cs::utils::CommandLine& commandLine, std::string* cLambdas, double* cMinLambda,
    double* cMaxLambda, int32_t* cLambdaSamples) {

  commandLine.addArgument({"--min-lambda"}, cMinLambda,
      "The minimum wavelength in µm (default: " + std::to_string(*cMinLambda) + ").");
  commandLine.addArgument({"--max-lambda"}, cMaxLambda,
      "The maximum wavelength in µm (default: " + std::to_string(*cMaxLambda) + ").");
  commandLine.addArgument({"--lambda-samples"}, cLambdaSamples,
      "The number of wavelengths to compute (default: " + std::to_string(*cLambdaSamples) + ").");
  commandLine.addArgument({"--lambdas"}, cLambdas,
      "A comma-separated list of wavelengths in µm. If provided, --min-lambda, --max-lambda, and "
      "--lambda-samples are ignored.");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::vector<double> computeLambdas(
    std::string const& cLambdas, double cMinLambda, double cMaxLambda, int32_t cLambdaSamples) {
  std::vector<double> lambdas;

  if (cLambdas.empty()) {
    if (cLambdaSamples <= 0) {
      std::cerr << "Lambda-sample count must be > 0!" << std::endl;
      return lambdas;
    } else if (cLambdaSamples == 1) {
      lambdas.push_back(cMinLambda);
    } else {
      for (int32_t i(0); i < cLambdaSamples; ++i) {
        lambdas.push_back(cMinLambda + (cMaxLambda - cMinLambda) * i / (cLambdaSamples - 1.0));
      }
    }
  } else {
    auto tokens = cs::utils::splitString(cLambdas, ',');
    for (auto token : tokens) {
      lambdas.push_back(cs::utils::fromString<double>(token));
    }
  }

  return lambdas;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace common
