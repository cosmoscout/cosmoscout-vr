////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSP_MEASUREMENT_TOOLS_LOGGER_HPP
#define CSP_MEASUREMENT_TOOLS_LOGGER_HPP

#include <spdlog/spdlog.h>

namespace csp::measurementtools {

/// This creates the default singleton logger for "csp-measurement-tools" when called for the first
/// time and returns it. See cs-utils/logger.hpp for more logging details.
spdlog::logger& logger();

} // namespace csp::measurementtools

#endif // CSP_MEASUREMENT_TOOLS_LOGGER_HPP
