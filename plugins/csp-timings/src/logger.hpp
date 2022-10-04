////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSP_TIMINGS_LOGGER_HPP
#define CSP_TIMINGS_LOGGER_HPP

#include <spdlog/spdlog.h>

namespace csp::timings {

/// This creates the default singleton logger for "csp-timings" when called for the first time
/// and returns it. See cs-utils/logger.hpp for more logging details.
spdlog::logger& logger();

} // namespace csp::timings

#endif // CSP_TIMINGS_LOGGER_HPP
