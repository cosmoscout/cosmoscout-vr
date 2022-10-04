////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSP_CUSTOM_WEB_UI_LOGGER_HPP
#define CSP_CUSTOM_WEB_UI_LOGGER_HPP

#include <spdlog/spdlog.h>

namespace csp::customwebui {

/// This creates the default singleton logger for "csp-custom-web-ui" when called for the first time
/// and returns it. See cs-utils/logger.hpp for more logging details.
spdlog::logger& logger();

} // namespace csp::customwebui

#endif // CSP_CUSTOM_WEB_UI_LOGGER_HPP
