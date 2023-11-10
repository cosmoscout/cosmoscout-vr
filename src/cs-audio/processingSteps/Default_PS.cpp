////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "Default_PS.hpp"
#include "../internal/alErrorHandling.hpp"
#include "../logger.hpp"

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

void Default_PS::process(std::shared_ptr<SourceBase> source, 
  std::shared_ptr<std::map<std::string, std::any>> settings,
  std::shared_ptr<std::vector<std::string>> failedSettings) {

  ALuint openAlId = source->getOpenAlId();
  
  if (auto search = settings->find("gain"); search != settings->end()) { 
    if (!processGain(openAlId, search->second)) {
      failedSettings->push_back("gain");
    }
  }

  if (auto search = settings->find("looping"); search != settings->end()) {
    if (!processLooping(openAlId, search->second)) {
      failedSettings->push_back("looping");
    }
  }

  if (auto search = settings->find("pitch"); search != settings->end()) {
    if (!processPitch(openAlId, search->second)) {
      failedSettings->push_back("pitch");
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool Default_PS::processGain(ALuint openAlId, std::any value) {
    if (value.type() != typeid(float)) {
      
      // remove gain
      if (value.type() == typeid(std::string) && std::any_cast<std::string>(value) == "remove") { 
        
        alSourcef(openAlId, AL_GAIN, 1.f);
        if (alErrorHandling::errorOccurred()) {
          logger().warn("Failed to reset source gain!");
          return false;
        }
        return true;
      }

      // wrong type provided
      logger().warn("Audio source settings error! Wrong type used for gain setting! Allowed Type: float");
      return false;
    }

    float floatValue = std::any_cast<float>(value);

    if (floatValue < 0.f) {
      logger().warn("Audio source settings error! Unable to set a negative gain!");
      return false;
    }

    alSourcef(openAlId, AL_GAIN, floatValue);

    if (alErrorHandling::errorOccurred()) {
      logger().warn("Failed to set source gain!");
      return false;
    }
    return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool Default_PS::processLooping(ALuint openAlId, std::any value) {
  if (value.type() != typeid(bool)) {

    // remove looping
    if (value.type() == typeid(std::string) && std::any_cast<std::string>(value) == "remove") { 
      std::cout << "looping removed" << std::endl;
      
      alSourcei(openAlId, AL_LOOPING, AL_FALSE);
      if (alErrorHandling::errorOccurred()) {
        logger().warn("Failed to reset source looping!");
        return false;
      }
      return true;
    }

    // wrong type provided
    logger().warn("Audio source settings error! Wrong type used for looping setting! Allowed Type: bool");
    return false;
  }

  bool boolValue = std::any_cast<bool>(value);

  alSourcei(openAlId, AL_LOOPING, boolValue);

  if (alErrorHandling::errorOccurred()) {
    logger().warn("Failed to set source looping!");
    return false;
  }
  return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool Default_PS::processPitch(ALuint openAlId, std::any value) {
  if (value.type() != typeid(float)) {

    // remove pitch
    if (value.type() == typeid(std::string) && std::any_cast<std::string>(value) == "remove") { 
      std::cout << "pitch removed" << std::endl;
      
      alSourcef(openAlId, AL_PITCH, 1.f);
      if (alErrorHandling::errorOccurred()) {
        logger().warn("Failed to reset source pitch!");
        return false;
      }
      return true;
    }

    // wrong type provided
    logger().warn("Audio source settings error! Wrong type used for pitch setting! Allowed Type: float");
    return false;
  }

  float floatValue = std::any_cast<float>(value);

  if (floatValue < 0.f) {
    logger().warn("Audio source error! Unable to set a negative pitch!");
    return false;
  }
  
  alSourcef(openAlId, AL_PITCH, floatValue);
  
  if (alErrorHandling::errorOccurred()) {
    logger().warn("Failed to set source pitch!");
    return false;
  }
  return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool Default_PS::requiresUpdate() const {
  return false;  
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Default_PS::update() {
  
}

} // namespace cs::audio