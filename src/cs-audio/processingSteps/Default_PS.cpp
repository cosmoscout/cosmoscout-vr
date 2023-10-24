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

std::shared_ptr<ProcessingStep> Default_PS::create() {
  static auto default_ps = std::shared_ptr<Default_PS>(new Default_PS());
  return default_ps;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

Default_PS::Default_PS() {}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Default_PS::process(ALuint openAlId, 
  std::shared_ptr<std::map<std::string, std::any>> settings,
  std::shared_ptr<std::vector<std::string>> failedSettings) {

  if (auto search = settings->find("gain"); search != settings->end()) { 
    if (!processGain(openAlId, settings->at("gain"))) {
      failedSettings->push_back("gain");
    }
  }

  if (auto search = settings->find("looping"); search != settings->end()) {
    if (!processLooping(openAlId, settings->at("looping"))) {
      failedSettings->push_back("looping");
    }
  }

  if (auto search = settings->find("pitch"); search != settings->end()) {
    if (!processPitch(openAlId, settings->at("pitch"))) {
      failedSettings->push_back("pitch");
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool Default_PS::processGain(ALuint openAlId, std::any value) {
    float floatValue;

    try {
      floatValue = std::any_cast<float>(value);
    } catch (const std::bad_any_cast&) {
      logger().warn("Audio source settings error! Wrong type used for gain setting! Allowed Type: float");
      return false;
    }

    if (floatValue < 0.f) {
      logger().warn("Audio source settings error! Unable to set a negative gain!");
      return false;
    
    } else {
      alSourcef(openAlId, AL_GAIN, floatValue);
      if (alErrorHandling::errorOccurred()) {
        logger().warn("Failed to set source gain!");
        return false;
      }
    }
    return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool Default_PS::processLooping(ALuint openAlId, std::any value) {
  bool boolValue;

  try {
    boolValue = std::any_cast<bool>(value);
  } catch (const std::bad_any_cast&) {
    logger().warn("Audio source settings error! Wrong type used for looping setting! Allowed Type: bool");
    return false;
  }

  alSourcei(openAlId, AL_LOOPING, boolValue);
  if (alErrorHandling::errorOccurred()) {
    logger().warn("Failed to set source looping!");
    return false;
  }
  return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool Default_PS::processPitch(ALuint openAlId, std::any value) {
  float floatValue;

  try {
    floatValue = std::any_cast<float>(value);
  } catch (const std::bad_any_cast&) {
    logger().warn("Audio source settings error! Wrong type used for pitch setting! Allowed Type: float");
    return false;
  }

  if (floatValue < 0.f) {
    logger().warn("Audio source error! Unable to set a negative pitch!");
    return false;

  } else {
    alSourcef(openAlId, AL_PITCH, floatValue);
    if (alErrorHandling::errorOccurred()) {
      logger().warn("Failed to set source pitch!");
      return false;
    }
  }
  return true;
}

} // namespace cs::audio