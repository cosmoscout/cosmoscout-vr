////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CS_SCENE_LOGGER_HPP
#define CS_SCENE_LOGGER_HPP

#include "cs_scene_export.hpp"

#include <spdlog/spdlog.h>

namespace cs::scene {

/// This creates the default singleton logger for "cs-scene" when called for the first time and
/// returns it. See cs-utils/logger.hpp for more logging details.
CS_SCENE_EXPORT spdlog::logger& logger();

} // namespace cs::scene

#endif // CS_SCENE_LOGGER_HPP
