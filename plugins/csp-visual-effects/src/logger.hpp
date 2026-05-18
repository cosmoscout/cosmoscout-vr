////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSP_VISUAL_EFFECTS_LOGGER_HPP
#define CSP_VISUAL_EFFECTS_LOGGER_HPP

#include <spdlog/spdlog.h>

namespace csp::visualeffects {

/// This creates the default singleton logger for "csp-visual-effects" when called for the first time
/// and returns it. See cs-utils/logger.hpp for more logging details.
spdlog::logger& logger();

} // namespace csp::visualeffects

#endif // CSP_VISUAL_EFFECTS_LOGGER_HPP
