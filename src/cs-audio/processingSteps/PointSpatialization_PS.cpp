////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "PointSpatialization_PS.hpp"
#include "../internal/alErrorHandling.hpp"
#include "../logger.hpp"

#include <AL/al.h>
#include <glm/detail/type_vec3.hpp>
#include <memory>
#include <vector>
#include <string>
#include <iostream>

#include <chrono>
#include <thread>

namespace cs::audio {

std::shared_ptr<ProcessingStep> PointSpatialization_PS::create() {
  static auto spatialization_ps = std::shared_ptr<PointSpatialization_PS>(new PointSpatialization_PS());
  return spatialization_ps;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

PointSpatialization_PS::PointSpatialization_PS()
  : mSourcePositions(std::map<ALuint, SourceContainer>())
  , mLastTime(std::chrono::system_clock::now()) {
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void PointSpatialization_PS::process(std::shared_ptr<Source> source, 
  std::shared_ptr<std::map<std::string, std::any>> settings,
  std::shared_ptr<std::vector<std::string>> failedSettings) {
  
  if (auto search = settings->find("position"); search != settings->end()) { 
    if (!processPosition(source, search->second)) {
      failedSettings->push_back("position");
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool PointSpatialization_PS::processPosition(std::shared_ptr<Source> source, std::any value) {
  
  if (value.type() != typeid(glm::dvec3)) {

    // remove position
    if (value.type() == typeid(std::string) && std::any_cast<std::string>(value) == "remove") { 

      ALuint openAlId = source->getOpenAlId();
      mSourcePositions.erase(openAlId);

      alSourcei(openAlId, AL_SOURCE_RELATIVE, AL_TRUE);
      if (alErrorHandling::errorOccurred()) {
        logger().warn("Failed to reset source position specification to relative!");
        return false;
      }
      
      alSource3f(openAlId, AL_POSITION, 
        (ALfloat)0.f, 
        (ALfloat)0.f, 
        (ALfloat)0.f);
      if (alErrorHandling::errorOccurred()) {
        logger().warn("Failed to reset source position!");
        return false;
      }

      return true;
    }

    logger().warn("Audio source settings error! Wrong type used for position setting! Allowed Type: glm::dvec3");
    return false;
  }

  ALuint openAlId = source->getOpenAlId();

  alSourcei(openAlId, AL_SOURCE_RELATIVE, AL_FALSE);
  if (alErrorHandling::errorOccurred()) {
    logger().warn("Failed to set source position specification to absolute!");
    return false;
  }

  glm::dvec3 positionValue = std::any_cast<glm::dvec3>(value);

  alSource3f(openAlId, AL_POSITION, 
    (ALfloat)positionValue.x, 
    (ALfloat)positionValue.y, 
    (ALfloat)positionValue.z);
  
  if (alErrorHandling::errorOccurred()) {
    logger().warn("Failed to set source position!");
    return false;
  }

  mSourcePositions[openAlId] = SourceContainer{std::weak_ptr<Source>(source), positionValue, positionValue};
  return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void PointSpatialization_PS::calculateVelocity() {
  std::chrono::system_clock::time_point currentTime = std::chrono::system_clock::now();
  std::chrono::duration<float> elapsed_seconds = currentTime - mLastTime; 
  auto elapsed_secondsf = elapsed_seconds.count();

  for (auto source : mSourcePositions) {
    
    if (source.second.sourcePtr.expired()) {
      mSourcePositions.erase(source.first);
      continue;
    }

    glm::dvec3 velocity;
    ALuint openAlId = source.second.sourcePtr.lock()->getOpenAlId(); 

    if (source.second.currentPos != source.second.lastPos) {
      glm::dvec3 posDelta = source.second.currentPos - source.second.lastPos;
      velocity.x = posDelta.x / elapsed_secondsf;
      velocity.y = posDelta.y / elapsed_secondsf;
      velocity.z = posDelta.z / elapsed_secondsf;
      mSourcePositions[openAlId].lastPos = source.second.currentPos;
      
    } else {  
      velocity.x = 0;
      velocity.y = 0;
      velocity.z = 0;
    }

    alSource3f(openAlId, AL_VELOCITY, 
      (ALfloat)velocity.x, 
      (ALfloat)velocity.y, 
      (ALfloat)velocity.z);

    if (alErrorHandling::errorOccurred()) {
      logger().warn("Failed to set source velocity!");
    }
  }

  mLastTime = currentTime;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool PointSpatialization_PS::requiresUpdate() const {
  return true;  
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void PointSpatialization_PS::update() {
  calculateVelocity();
}

} // namespace cs::audio