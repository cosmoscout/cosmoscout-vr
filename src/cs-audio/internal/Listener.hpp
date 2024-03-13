////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CS_AUDIO_LISTENER_HPP
#define CS_AUDIO_LISTENER_HPP

#include "cs_audio_export.hpp"

namespace cs::audio {

/// @brief This class offers controls to adjust the OpenAL Listeners transformation. These
/// transformation should only affect spatialized sources (all others are supposed
/// to have a relative position of (0,0,0) to the listener and are therefor not affected).
/// These transformations are, in the current version of December 2023, not used.
class CS_AUDIO_EXPORT Listener {
 public:
  /// @brief Sets the OpenAL Listener position
  /// @return True if no error occurred
  static bool setPosition(float x, float y, float z);

  /// @brief Sets the OpenAL Listener velocity
  /// @return True if no error occurred
  static bool setVelocity(float x, float y, float z);

  /// @brief Sets the OpenAL Listener orientation.
  /// @return True if no error occurred
  static bool setOrientation(float atX, float atY, float atZ, float upX, float upY, float upZ);
};

} // namespace cs::audio

#endif // CS_AUDIO_LISTENER_HPP
