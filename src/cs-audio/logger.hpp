////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CS_AUDIO_LOGGER_HPP
#define CS_AUDIO_LOGGER_HPP

#include "cs_audio_export.hpp"

#include <spdlog/spdlog.h>

namespace cs::audio {

/// This creates the default singleton logger for "cs-audio" when called for the first time and
/// returns it. See cs-utils/logger.hpp for more logging details.
CS_AUDIO_EXPORT spdlog::logger& logger();

} // namespace cs::audio

#endif // CS_AUDIO_LOGGER_HPP
