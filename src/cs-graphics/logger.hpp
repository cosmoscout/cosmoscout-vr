////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CS_GRAPHICS_LOGGER_HPP
#define CS_GRAPHICS_LOGGER_HPP

#include "cs_graphics_export.hpp"

namespace cs::graphics::logger {

/// This creates the default logger for "cs-graphics" and is called at startup by the main() method.
/// See ../cs-utils/logger.hpp for more logging details.
CS_GRAPHICS_EXPORT void init();

} // namespace cs::graphics::logger

#endif // CS_GRAPHICS_LOGGER_HPP
