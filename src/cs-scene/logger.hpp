////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CS_SCENE_LOGGER_HPP
#define CS_SCENE_LOGGER_HPP

#include "cs_scene_export.hpp"

namespace cs::scene::logger {

/// This creates the default logger for "cs-scene" and is called at startup by the main() method.
/// See ../cs-utils/logger.hpp for more logging details.
CS_SCENE_EXPORT void init();

} // namespace cs::scene::logger

#endif // CS_SCENE_LOGGER_HPP
