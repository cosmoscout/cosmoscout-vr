////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "SphereSpatialization_PS.hpp"
#include "../internal/alErrorHandling.hpp"
#include "../logger.hpp"
#include <cmath>
#include <glm/detail/type_vec3.hpp>
#include <glm/glm.hpp>
#include <glm/fwd.hpp>

namespace cs::audio {
// TODO: sourceRadius muss >0 sein 
std::shared_ptr<ProcessingStep> SphereSpatialization_PS::create() {
  static auto sphereSpatialization_PS = 
    std::shared_ptr<SphereSpatialization_PS>(new SphereSpatialization_PS());
  return sphereSpatialization_PS;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

SphereSpatialization_PS::SphereSpatialization_PS() {}

////////////////////////////////////////////////////////////////////////////////////////////////////

void SphereSpatialization_PS::process(std::shared_ptr<SourceBase> source, 
  std::shared_ptr<std::map<std::string, std::any>> settings,
  std::shared_ptr<std::vector<std::string>> failedSettings) {
  
  ALuint openAlId = source->getOpenAlId();
  bool processRequired = false;
  std::any pos, radius;
  if (auto searchPos = settings->find("position"); searchPos != settings->end()) {
    if (processPosition(openAlId, searchPos->second)) {
      processRequired = true;
      pos = searchPos->second;
    } else {
        failedSettings->push_back("position");
    }
  }

  if (auto searchRad = settings->find("sourceRadius"); searchRad != settings->end()) {
    if (processRadius(openAlId, searchRad->second)) {
      processRequired = true;
      radius = searchRad->second;
    } else {
        failedSettings->push_back("sourceRadius");
      }
    }

  if (processRequired) {
    if (!pos.has_value()) {
      auto currentSettings = source->getPlaybackSettings();
      if (auto searchPos = currentSettings->find("position"); searchPos != currentSettings->end()) {
        pos = searchPos->second;
      } else {
        return;
      }
    }

    if (!radius.has_value()) {
      auto currentSettings = source->getPlaybackSettings();
      if (auto searchRad = currentSettings->find("sourceRadius"); searchRad != currentSettings->end()) {
        radius = searchRad->second;
      } else {
        return;
      }
    }

    processSpatialization(source, pos, radius);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool SphereSpatialization_PS::processPosition(ALuint openAlId, std::any position) {
  if (position.type() != typeid(glm::dvec3)) {

    // remove position setting from source
    if (position.type() == typeid(std::string) && std::any_cast<std::string>(position) == "remove") {
      return resetSpatialization(openAlId);      
    }

    // wrong datatype used for position 
    logger().warn("Audio source settings error! Wrong type used for position setting! Allowed Type: glm::dvec3");
    return false;
  }
  return true;
}

bool SphereSpatialization_PS::processRadius(ALuint openAlId, std::any sourceRadius) {
  if (sourceRadius.type() != typeid(double)) {
    
    // remove source radius setting from source
    if (sourceRadius.type() == typeid(std::string) && std::any_cast<std::string>(sourceRadius) == "remove") {
      return resetSpatialization(openAlId);
    }

    // wrong datatype used for position 
    logger().warn("Audio source settings error! Wrong type used for sourceRadius setting! Allowed Type: double");
    return false;
  }
  return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool SphereSpatialization_PS::processSpatialization(std::shared_ptr<SourceBase> source, 
  std::any position ,std::any sourceRadius) {
    
  auto sourcePosToObserver = std::any_cast<glm::dvec3>(position);
  rotateSourcePosByViewer(sourcePosToObserver);
  auto radius = std::any_cast<double>(sourceRadius);
  ALuint openAlId = source->getOpenAlId();

  // Set source position to Observer Pos if the Observer is inside the source radius.
  // Otherwise set to the real position.
  alSourcei(openAlId, AL_SOURCE_RELATIVE, AL_FALSE);
  if (alErrorHandling::errorOccurred()) {
    logger().warn("Failed to set source position specification to absolute!");
    return false;
  }

  if (glm::length(sourcePosToObserver) < radius) {
    alSource3f(openAlId, AL_POSITION, 
      (ALfloat)0.f, 
      (ALfloat)0.f, 
      (ALfloat)0.f);
    if (alErrorHandling::errorOccurred()) {
      logger().warn("Failed to set source position!");
      return false;
    }

  } else {
    alSource3f(openAlId, AL_POSITION, 
      (ALfloat)sourcePosToObserver.x, 
      (ALfloat)sourcePosToObserver.y, 
      (ALfloat)sourcePosToObserver.z);
    if (alErrorHandling::errorOccurred()) {
      logger().warn("Failed to set source position!");
      return false;
    }
  }

  mSourcePositions[openAlId] = 
    SourceContainer{std::weak_ptr<SourceBase>(source), sourcePosToObserver, sourcePosToObserver};

  return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool SphereSpatialization_PS::resetSpatialization(ALuint openAlId) {
  // TODO: erase source from list
  alSource3f(openAlId, AL_POSITION, 
    (ALfloat)0.f, 
    (ALfloat)0.f, 
    (ALfloat)0.f);
  if (alErrorHandling::errorOccurred()) {
    logger().warn("Failed to reset source position!");
    return false;
  }

  alSource3f(openAlId, AL_VELOCITY, 
    (ALfloat)0.f, 
    (ALfloat)0.f, 
    (ALfloat)0.f);
  if (alErrorHandling::errorOccurred()) {
    logger().warn("Failed to reset source velocity!");
    return false;
  }
  return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool SphereSpatialization_PS::requiresUpdate() const {
  return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void SphereSpatialization_PS::update() {
  calculateVelocity();
}

void ScaledSphereSpatialization_PS::calculateVelocity() {
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

} // namespace cs::audio