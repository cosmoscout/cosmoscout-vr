////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "DistanceCulling_PS.hpp"
#include "../internal/AlErrorHandling.hpp"
#include "../logger.hpp"

#include <AL/al.h>
#include <map>
#include <glm/fwd.hpp>
#include <glm/glm.hpp>
#include <any>

namespace cs::audio {

std::shared_ptr<ProcessingStep> DistanceCulling_PS::create(double distanceThreshold) {
  static auto distanceCulling_ps = std::shared_ptr<DistanceCulling_PS>(new DistanceCulling_PS(distanceThreshold));
  return distanceCulling_ps;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

DistanceCulling_PS::DistanceCulling_PS(double distanceThreshold)
  : mDistanceThreshold(std::move(distanceThreshold)) {
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void DistanceCulling_PS::process(std::shared_ptr<SourceBase> source, 
  std::shared_ptr<std::map<std::string, std::any>> settings,
  std::shared_ptr<std::vector<std::string>> failedSettings) {
  
  if (auto searchPos = settings->find("position"); searchPos != settings->end()) { 

    auto searchPlayback = settings->find("playback");
    std::any newPlayback = (searchPlayback != settings->end() ? searchPlayback->second : std::any());

    if (!processPosition(source, searchPos->second, newPlayback)) {
      failedSettings->push_back("playback");
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool DistanceCulling_PS::processPosition(std::shared_ptr<SourceBase> source, std::any position,
  std::any newPlayback) {

  ALuint openALId = source->getOpenAlId();

  // validate position setting
  if (position.type() != typeid(glm::dvec3)) {
    // remove position
    if (position.type() == typeid(std::string) && std::any_cast<std::string>(position) == "remove") { 

      alSourceStop(openALId);
      if (AlErrorHandling::errorOccurred()) {
        logger().warn("Failed to stop source playback!");
        return false;
      }
      return true;
    }

    // wrong type passed
    logger().warn("Audio source settings error! Wrong type used for position setting! Allowed Type: glm::dvec3");
    return false;
  }

  // Search for the currently set playback state(play, stop, pause)
  std::string supposedState;
  if (newPlayback.has_value()) {
    if (newPlayback.type() == typeid(std::string)) {
      supposedState = std::any_cast<std::string>(newPlayback);
    }
  
  } else {
    auto settings = source->getPlaybackSettings();
    if (auto search = settings->find("playback"); search != settings->end()) {
      supposedState = std::any_cast<std::string>(search->second);
    }
  }

  // Get currently set state in OpenAL
  ALint isState;
  alGetSourcei(openALId, AL_SOURCE_STATE, &isState);

  /* Evaluate what to do based on the supposedState and isState of a source. Possible combinations:

  supposedState   isState    do 
  -------------------------------------------
  play            play       compute culling
                  stop       compute culling
                  pause      compute culling
  stop            play       stop playback
                  stop       nothing
                  pause      stop playback
  pause           play       pause playback
                  stop       pause playback
                  pause      nothing
  */

  if (supposedState == "stop") {
    switch(isState) {

      case AL_PLAYING:
        alSourceStop(openALId);
        if (AlErrorHandling::errorOccurred()) {
          logger().warn("Failed to stop playback of source!");
          return false;
        }
        return true;

      case AL_PAUSED:
        alSourceStop(openALId);
        if (AlErrorHandling::errorOccurred()) {
          logger().warn("Failed to stop playback of source!");
          return false;
        }
        return true;

      case AL_STOPPED:
      default:
        return true;
    }
  }

  if (supposedState == "pause") {
    switch(isState) {

      case AL_PLAYING:
        alSourcePause(openALId);
        if (AlErrorHandling::errorOccurred()) {
          logger().warn("Failed to pause playback of source!");
          return false;
        }
        return true;

      case AL_STOPPED:
        alSourcePause(openALId);
        if (AlErrorHandling::errorOccurred()) {
          logger().warn("Failed to pause playback of source!");
          return false;
        }
        return true;
      
      case AL_PAUSED:
      default:
        return true;
    }
  }

  // compute culling:
  if (supposedState == "play") {

    glm::dvec3 sourcePosToObserver = std::any_cast<glm::dvec3>(position);
    double distance = glm::length(sourcePosToObserver);
    
    // start/pause source based on the distance compared to the specified threshold
    if (distance > mDistanceThreshold) {
      if (isState != AL_PAUSED && isState != AL_INITIAL) {
        alSourcePause(openALId);
        if (AlErrorHandling::errorOccurred()) {
          logger().warn("Failed to pause playback of source!");
          return false;
        }
      }
    } else {
      if (isState != AL_PLAYING) {
        alSourcePlay(openALId);
        if (AlErrorHandling::errorOccurred()) {
          logger().warn("Failed to start playback of source!");
          return false;
        }
      }
    }
    return true;
  }

  logger().warn("Unkown value passed to playback settings! Allowed values: 'play', 'pause', 'stop'");
  return false;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool DistanceCulling_PS::requiresUpdate() const {
  return false;  
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void DistanceCulling_PS::update() {
    
}

} // namespace cs::audio