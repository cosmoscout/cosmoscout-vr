////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "AlErrorHandling.hpp"
#include "../logger.hpp"

namespace cs::audio {

bool AlErrorHandling::errorOccurred() {
  ALenum error;
  if ((error = alGetError()) != AL_NO_ERROR) {

    std::string errorCode;
    switch (error) {
    case AL_INVALID_NAME:
      errorCode = "Invalid name (ID) passed to an AL call";
      break;
    case AL_INVALID_ENUM:
      errorCode = "Invalid enumeration passed to AL call";
      break;
    case AL_INVALID_VALUE:
      errorCode = "Invalid value passed to AL call";
      break;
    case AL_INVALID_OPERATION:
      errorCode = "Illegal AL call";
      break;
    case AL_OUT_OF_MEMORY:
      errorCode = "Not enough memory to execute the AL call";
      break;
    default:
      errorCode = "Unkown error code";
    }

    logger().warn("OpenAL-Soft Error occured! Reason: {}...", errorCode);
    return true;
  }
  return false;
}

} // namespace cs::audio