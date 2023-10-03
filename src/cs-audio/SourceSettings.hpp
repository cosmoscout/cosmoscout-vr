////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CS_CORE_AUDIO_SOURCE_SETTINGS_HPP
#define CS_CORE_AUDIO_SOURCE_SETTINGS_HPP

#include "cs_audio_export.hpp"

#include <optional>

namespace cs::audio {

struct Vec3 {
  float x;
  float y;
  float z;
};

struct CS_AUDIO_EXPORT SourceSettings {
  std::optional<float> gain;
  std::optional<float> pitch;
  std::optional<bool> looping;
};

} // namespace cs::audio

#endif // CS_CORE_AUDIO_SOURCE_SETTINGS_HPP
