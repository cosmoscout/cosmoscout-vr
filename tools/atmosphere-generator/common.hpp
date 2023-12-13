////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef COMMON_HPP
#define COMMON_HPP

#include "../../src/cs-utils/CommandLine.hpp"

////////////////////////////////////////////////////////////////////////////////////////////////////
// This method writes ozone absorption cross-sections in the same format as the mie mode.         //
////////////////////////////////////////////////////////////////////////////////////////////////////

namespace common {
void addLambdaFlags(cs::utils::CommandLine& commandLine, std::string* cLambdas, double* cMinLambda,
    double* cMaxLambda, int32_t* cLambdaSamples);

std::vector<double> computeLambdas(
    std::string const& cLambdas, double cMinLambda, double cMaxLambda, int32_t cLambdaSamples);
} // namespace common

#endif // COMMON_HPP