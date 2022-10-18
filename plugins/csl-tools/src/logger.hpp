////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSL_TOOLS_LOGGER_HPP
#define CSL_TOOLS_LOGGER_HPP

#include <spdlog/spdlog.h>

namespace csl::tools {

/// This creates the default singleton logger for "csl-measurement-tools-base" when called for the
/// first time and returns it. See cs-utils/logger.hpp for more logging details.
spdlog::logger& logger();

} // namespace csl::tools

#endif // CSL_TOOLS_LOGGER_HPP
