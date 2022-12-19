////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSP_ADVANCED_ATMOSPHERES_LOGGER_HPP
#define CSP_ADVANCED_ATMOSPHERES_LOGGER_HPP

#include <spdlog/spdlog.h>

namespace csp::advanced_atmospheres {

/// This creates the default singleton logger for "csp-atmospheres" when called for the first time
/// and returns it. See cs-utils/logger.hpp for more logging details.
spdlog::logger& logger();

} // namespace csp::advanced_atmospheres

#endif // CSP_ADVANCED_ATMOSPHERES_LOGGER_HPP
