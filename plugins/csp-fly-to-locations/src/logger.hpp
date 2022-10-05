////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSP_FLY_TO_LOCATIONS_LOGGER_HPP
#define CSP_FLY_TO_LOCATIONS_LOGGER_HPP

#include <spdlog/spdlog.h>

namespace csp::flytolocations {

/// This creates the default singleton logger for "csp-fly-to-locations" when called for the first
/// time and returns it. See cs-utils/logger.hpp for more logging details.
spdlog::logger& logger();

} // namespace csp::flytolocations

#endif // CSP_FLY_TO_LOCATIONS_LOGGER_HPP
