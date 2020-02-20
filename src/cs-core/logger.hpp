////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CS_CORE_LOGGER_HPP
#define CS_CORE_LOGGER_HPP

#include "cs_core_export.hpp"

namespace cs::core::logger {

/// This creates the default logger for "cs-core" and is called at startup by the main() method.
/// See ../cs-utils/logger.hpp for more logging details.
CS_CORE_EXPORT void init();

} // namespace cs::core::logger

#endif // CS_CORE_LOGGER_HPP
