////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

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
