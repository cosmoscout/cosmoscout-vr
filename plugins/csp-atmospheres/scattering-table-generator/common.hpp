////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef COMMON_HPP
#define COMMON_HPP

#include "../../../src/cs-utils/CommandLine.hpp"

////////////////////////////////////////////////////////////////////////////////////////////////////
// Common functionality which is used by multiple modes.                                          //
////////////////////////////////////////////////////////////////////////////////////////////////////

namespace common {

/// This adds the --lambda-samples, --min-lambda, --max-lambda, and --lambda-samples commandline
/// parameters to the given CommandLine object.
void addLambdaFlags(cs::utils::CommandLine& commandLine, std::string* lambdas, double* minLambda,
    double* maxLambda, int32_t* lambdaSamples);

/// This adds the --theta-samples commandline parameter to the given CommandLine object.
void addThetaFlags(cs::utils::CommandLine& commandLine, int32_t* thetaSamples);

/// This assembles a list of wavelengths from the given parameters. This is either provided with the
/// comma-separated 'lambdas' string, or via the combination of minLambda, maxLambda, and
/// lambdaSamples.
std::vector<double> computeLambdas(
    std::string const& lambdas, double minLambda, double maxLambda, int32_t lambdaSamples);

/// Converts a comma-separated string of numbers to a vector of doubles.
std::vector<double> parseNumberList(std::string const& list);

/// This samples a piece-wise linear function at the given xValue. Y values are given for evenly
/// spaced X values between xMin and xMax.
double interpolate(std::vector<double> yValues, double xMin, double xMax, double xValue);

} // namespace common

#endif // COMMON_HPP