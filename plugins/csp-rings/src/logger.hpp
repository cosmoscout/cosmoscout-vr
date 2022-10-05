////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSP_RINGS_LOGGER_HPP
#define CSP_RINGS_LOGGER_HPP

#include <spdlog/spdlog.h>

namespace csp::rings {

/// This creates the default singleton logger for "csp-rings" when called for the first time
/// and returns it. See cs-utils/logger.hpp for more logging details.
spdlog::logger& logger();

} // namespace csp::rings

#endif // CSP_RINGS_LOGGER_HPP
