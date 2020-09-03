////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CS_LOGGER_HPP
#define CS_LOGGER_HPP

#include <spdlog/spdlog.h>

/// This creates the default singleton logger for "cosmoscout-vr" when called for the first time and
/// returns it. See cs-utils/logger.hpp for more logging details.
spdlog::logger& logger();

#endif // CS_LOGGER_HPP
