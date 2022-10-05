////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSP_RECORDER_LOGGER_HPP
#define CSP_RECORDER_LOGGER_HPP

#include <spdlog/spdlog.h>

namespace csp::recorder {

/// This creates the default singleton logger for "csp-recorder" when called for the first time
/// and returns it. See cs-utils/logger.hpp for more logging details.
spdlog::logger& logger();

} // namespace csp::recorder

#endif // CSP_RECORDER_LOGGER_HPP
