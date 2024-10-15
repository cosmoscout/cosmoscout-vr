////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSL_OGC_LOGGER_HPP
#define CSL_OGC_LOGGER_HPP

#include <spdlog/spdlog.h>

namespace csl::ogc {

/// This creates the default singleton logger for "csl-ogc" when called for the first time
/// and returns it. See cs-utils/logger.hpp for more logging details.
spdlog::logger& logger();

} // namespace csl::ogc

#endif // CSL_OGC_LOGGER_HPP
