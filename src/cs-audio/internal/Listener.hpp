////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CS_AUDIO_LISTENER_HPP
#define CS_AUDIO_LISTENER_HPP

#include "cs_audio_export.hpp"
#include "BufferManager.hpp"
#include "OpenAlError.hpp"

#include <AL/al.h>

// forward declaration
class AudioEngine;

namespace cs::audio {

class CS_AUDIO_EXPORT Listener : public OpenAlError {
 public:
 
  static bool setPosition(float x, float y, float z);
  static bool setVeclocity(float x, float y, float z);
  static bool setOrientation(float atX, float atY, float atZ, float upX, float upY, float upZ);

 private:

};

} // namespace cs::audio

#endif // CS_AUDIO_LISTENER_HPP
