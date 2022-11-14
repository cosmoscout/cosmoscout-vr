////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSP_WEB_API_LOGGER_HPP
#define CSP_WEB_API_LOGGER_HPP

#include <spdlog/spdlog.h>

namespace csp::demonodeeditor {

/// This creates the default singleton logger for "csp-demonodeeditor" when called for the first
/// time and returns it. See cs-utils/logger.hpp for more logging details.
spdlog::logger& logger();

} // namespace csp::demonodeeditor

#endif // CSP_WEB_API_LOGGER_HPP
