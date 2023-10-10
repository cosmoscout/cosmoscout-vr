////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "Default_PS.hpp"
#include "../internal/alErrorHandling.hpp"

#include <AL/al.h>
#include <map>
#include <any>

namespace cs::audio {

void Default_PS::process(ALuint openAlId, std::shared_ptr<std::map<std::string, std::any>> settings) {
  if (auto search = settings->find("gain"); search != settings->end()) { 
    if (std::any_cast<float>(settings->at("gain")) < 0) {
      logger().warn("Audio source error! Unable to set a negative gain!");
    
    } else {
      alSourcef(openAlId, AL_GAIN, std::any_cast<float>(settings->at("gain")));
      if (alErrorHandling::errorOccurred()) {
          logger().warn("Failed to set source gain!");
      }
    }
  }

  if (auto search = settings->find("looping"); search != settings->end()) {
    alSourcei(openAlId, AL_LOOPING, std::any_cast<bool>(settings->at("looping")));
    if (alErrorHandling::errorOccurred()) {
      logger().warn("Failed to set source looping!");
    }
  }

  if (auto search = settings->find("pitch"); search != settings->end()) {
    if (std::any_cast<float>(settings->at("pitch")) < 0) {
      logger().warn("Audio source error! Unable to set a negative pitch!");
    
    } else {
      alSourcef(openAlId, AL_PITCH, std::any_cast<float>(settings->at("pitch")));
      if (alErrorHandling::errorOccurred()) {
        logger().warn("Failed to set source pitch!");
      }
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cs::audio