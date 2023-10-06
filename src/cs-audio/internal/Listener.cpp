////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "alErrorHandling.hpp"
#include "Listener.hpp"
#include "../logger.hpp"

#include <AL/al.h>

namespace cs::audio {

bool Listener::setPosition(float x, float y, float z) {
  alGetError(); // clear error code
  alListener3f(AL_POSITION, x, y, z);
  if (alErrorHandling::errorOccurd()) {
    logger().warn("Failed to set Listener Position!");
    return false;
  }
  return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool Listener::setVeclocity(float x, float y, float z) {
  alGetError(); // clear error code
  alListener3f(AL_VELOCITY, x, y, z);
  if (alErrorHandling::errorOccurd()) {
    logger().warn("Failed to set Listener Veclocity!");
    return false;
  }
  return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool Listener::setOrientation(float atX, float atY, float atZ, float upX, float upY, float upZ) {
  alGetError(); // clear error code
  ALfloat vec[] = { atX, atY, atZ, upX, upY, upZ };
  alListenerfv(AL_ORIENTATION, vec);
  if (alErrorHandling::errorOccurd()) {
    logger().warn("Failed to set Listener Veclocity!");
    return false;
  }
  return true;
}

} // namespace cs::audio