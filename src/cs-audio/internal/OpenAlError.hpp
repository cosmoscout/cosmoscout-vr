////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CS_AUDIO_ERROR_HANDLING_HPP
#define CS_AUDIO_ERROR_HANDLING_HPP

#include "cs_audio_export.hpp"

#include <AL/al.h>

// forward declaration
class AudioEngine;

namespace cs::audio {

class CS_AUDIO_EXPORT OpenAlError {
 protected:
  static bool errorOccurd();

}; // namespace cs::audio

} // cs::audio

#endif // CS_AUDIO_ERROR_HANDLING_HPP
