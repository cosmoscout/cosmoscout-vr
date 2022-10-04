////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CS_CORE_LOGGER_HPP
#define CS_CORE_LOGGER_HPP

#include "cs_core_export.hpp"

#include <spdlog/spdlog.h>

namespace cs::core {

/// This creates the default singleton logger for "cs-core" when called for the first time and
/// returns it. See cs-utils/logger.hpp for more logging details.
CS_CORE_EXPORT spdlog::logger& logger();

} // namespace cs::core

#endif // CS_CORE_LOGGER_HPP
