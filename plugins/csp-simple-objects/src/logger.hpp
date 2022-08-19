////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2022 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CSP_SIMPLE_OBJECTS_LOGGER_HPP
#define CSP_SIMPLE_OBJECTS_LOGGER_HPP

#include <spdlog/spdlog.h>

namespace csp::simpleobjects {

/// This creates the default singleton logger for "csp-simple-objects" when called for the first time
/// and returns it. See cs-utils/logger.hpp for more logging details.
spdlog::logger& logger();

} // namespace csp::simpleobjects

#endif // CSP_SIMPLE_OBJECTS_LOGGER_HPP
