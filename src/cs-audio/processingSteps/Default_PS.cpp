////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "Default_PS.hpp"
#include "../internal/alErrorHandling.hpp"
#include "../SourceSettings.hpp"

#include <AL/al.h>

namespace cs::audio {

void Default_PS::process(ALuint openAlId, std::shared_ptr<SourceSettings> settings) {
  if (settings->gain.has_value()) { 
    if (settings->gain.value() < 0) {
      logger().warn("Audio source error! Unable to set a negative gain!");
    
    } else {
      alSourcef(openAlId, AL_GAIN, settings->gain.value());
      if (alErrorHandling::errorOccurd()) {
          logger().warn("Failed to set source gain!");
      }
    }
  }

  if (settings->looping.has_value()) {
    alSourcei(openAlId, AL_LOOPING, settings->looping.value());
    if (alErrorHandling::errorOccurd()) {
      logger().warn("Failed to set source looping!");
    }
  }

  if (settings->pitch.has_value()) {
    if (settings->pitch.value() < 0) {
      logger().warn("Audio source error! Unable to set a negative pitch!");
    
    } else {
      alSourcef(openAlId, AL_PITCH, settings->pitch.value());
      if (alErrorHandling::errorOccurd()) {
        logger().warn("Failed to set source pitch!");
      }
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cs::audio