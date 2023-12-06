////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "DirectPlay_PS.hpp"
#include "../internal/AlErrorHandling.hpp"
#include "../logger.hpp"

#include <AL/al.h>
#include <any>

namespace cs::audio {

std::shared_ptr<ProcessingStep> DirectPlay_PS::create() {
  static auto directPlay_PS = std::shared_ptr<DirectPlay_PS>(new DirectPlay_PS());
  return directPlay_PS;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

DirectPlay_PS::DirectPlay_PS() {}

////////////////////////////////////////////////////////////////////////////////////////////////////

void DirectPlay_PS::process(std::shared_ptr<SourceBase> source, 
  std::shared_ptr<std::map<std::string, std::any>> settings,
  std::shared_ptr<std::vector<std::string>> failedSettings) {

  ALuint openAlId = source->getOpenAlId();
  
  if (auto search = settings->find("playback"); search != settings->end()) { 
    if (!processPlayback(openAlId, search->second)) {
      failedSettings->push_back("playback");
    }
  }
}

bool DirectPlay_PS::processPlayback(ALuint openAlId, std::any value) {
  // wrong type passed
  if (value.type() != typeid(std::string)) {
    logger().warn("Audio source settings error! Wrong type used for playback setting! Allowed Type: std::string");
    return false;
  }

  std::string stringValue = std::any_cast<std::string>(value);
  
  if (stringValue == "play") {
    alSourcePlay(openAlId);
    if (AlErrorHandling::errorOccurred()) {
      logger().warn("Failed to start playback of source!");
      return false;
    }
    return true;
  }

  else if (stringValue == "stop") {
    alSourceStop(openAlId);
    if (AlErrorHandling::errorOccurred()) {
      logger().warn("Failed to stop playback of source!");
      return false;
    }
    return true;
  }

  else if (stringValue == "pause") {
    alSourcePause(openAlId);
    if (AlErrorHandling::errorOccurred()) {
      logger().warn("Failed to pause playback of source!");
      return false;
    }
    return true;
  }
  logger().warn("Unkown value passed to playback settings! Allowed values: 'play', 'pause', 'stop'");
  return false;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool DirectPlay_PS::requiresUpdate() const {
  return false;  
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void DirectPlay_PS::update() {
  
}

} // namespace cs::audio