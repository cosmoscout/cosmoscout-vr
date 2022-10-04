////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CS_GRAPHICS_LOGGER_HPP
#define CS_GRAPHICS_LOGGER_HPP

#include "cs_graphics_export.hpp"

#include <spdlog/spdlog.h>

namespace cs::graphics {

/// This creates the default singleton logger for "cs-graphics" when called for the first time and
/// returns it. See cs-utils/logger.hpp for more logging details.
CS_GRAPHICS_EXPORT spdlog::logger& logger();

} // namespace cs::graphics

#endif // CS_GRAPHICS_LOGGER_HPP
