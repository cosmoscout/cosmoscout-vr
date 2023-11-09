////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "VolumeCulling_PS.hpp"
#include "../internal/alErrorHandling.hpp"
#include "../logger.hpp"

#include <AL/al.h>
#include <map>

namespace cs::audio {

std::shared_ptr<ProcessingStep> VolumeCulling_PS::create(float gainThreshold) {
  static auto volumeCulling_ps = std::shared_ptr<VolumeCulling_PS>(new VolumeCulling_PS(gainThreshold));
  return volumeCulling_ps;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

VolumeCulling_PS::VolumeCulling_PS(float gainThreshold)
  : mGainThreshold(std::move(gainThreshold)) {
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void VolumeCulling_PS::process(std::shared_ptr<Source> source, 
  std::shared_ptr<std::map<std::string, std::any>> settings,
  std::shared_ptr<std::vector<std::string>> failedSettings) {
  
  if (auto searchPos = settings->find("position"); searchPos != settings->end()) { 

    auto searchGain = settings->find("gain");
    std::any newGain = (searchGain != settings->end() ? searchGain->second : std::any());

    auto searchPlayback = settings->find("playback");
    std::any newPlayback = (searchPlayback != settings->end() ? searchPlayback->second : std::any());

    if (!processPosition(source, searchPos->second, newGain, newPlayback)) {
      failedSettings->push_back("playback");
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool VolumeCulling_PS::processPosition(std::shared_ptr<Source> source, std::any position, 
  std::any newGain, std::any newPlayback) {

  ALuint openALId = source->getOpenAlId();

  // validate position setting
  if (position.type() != typeid(glm::dvec3)) {
    // remove position
    if (position.type() == typeid(std::string) && std::any_cast<std::string>(position) == "remove") { 

      alSourceStop(openALId);
      if (alErrorHandling::errorOccurred()) {
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
        if (alErrorHandling::errorOccurred()) {
          logger().warn("Failed to stop playback of source!");
          return false;
        }
        return true;

      case AL_PAUSED:
        alSourceStop(openALId);
        if (alErrorHandling::errorOccurred()) {
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
        if (alErrorHandling::errorOccurred()) {
          logger().warn("Failed to pause playback of source!");
          return false;
        }
        return true;

      case AL_STOPPED:
        alSourcePause(openALId);
        if (alErrorHandling::errorOccurred()) {
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

    ALint disModel;
    ALfloat rollOffFac, refDis, maxDis;
    alGetSourcei(openALId, AL_DISTANCE_MODEL, &disModel);
    alGetSourcef(openALId, AL_ROLLOFF_FACTOR, &rollOffFac);
    alGetSourcef(openALId, AL_REFERENCE_DISTANCE, &refDis);
    alGetSourcef(openALId, AL_MAX_DISTANCE, &maxDis);

    double distance = glm::length(sourcePosToObserver);
    distance = (distance > refDis ? distance : refDis);
    distance = (distance < maxDis ? distance : maxDis);

    double supposedVolume;
    switch (disModel) {
      case AL_INVERSE_DISTANCE_CLAMPED:
        supposedVolume = inverseClamped(distance, rollOffFac, refDis, maxDis);
        break;
      case AL_LINEAR_DISTANCE_CLAMPED:
        supposedVolume = linearClamped(distance, rollOffFac, refDis, maxDis);
        break;
      case AL_EXPONENT_DISTANCE_CLAMPED:
        supposedVolume = exponentClamped(distance, rollOffFac, refDis, maxDis);
        break;
      default:
        logger().warn("Unsupported distance model used! Only clamped distance models are supported!");
        return false;
    }

    // Multiply just calculated volume with the gain of the source:
    float gain = -1.f;
    // check if a new gain is being set during this update cycle 
    if (newGain.has_value()) {
      if (newGain.type() == typeid(float)) {
        gain = std::any_cast<float>(newGain);
      }

    // else check if a gain was set in a previous update cycle
    } else {
      auto settings = source->getPlaybackSettings();
      if (auto search = settings->find("gain"); search != settings->end()) {
        gain = std::any_cast<float>(search->second);
      }
    }

    if (gain != -1.f) {
      supposedVolume *= gain;
    }

    // start/pause source based on the volume compared to the specified threshold
    if (supposedVolume < mGainThreshold) {
      if (isState != AL_PAUSED && isState != AL_INITIAL) {
        alSourcePause(openALId);
        if (alErrorHandling::errorOccurred()) {
          logger().warn("Failed to pause playback of source!");
          return false;
        }
      }
    } else {
      if (isState != AL_PLAYING) {
        alSourcePlay(openALId);
        if (alErrorHandling::errorOccurred()) {
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

double VolumeCulling_PS::inverseClamped(double distance, ALfloat rollOffFactor, 
  ALfloat referenceDistance, ALfloat maxDistance) {
  return 
    referenceDistance / (referenceDistance + rollOffFactor * (distance - referenceDistance));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

double VolumeCulling_PS::linearClamped(double distance, ALfloat rollOffFactor,
  ALfloat referenceDistance, ALfloat maxDistance) {
  return 
    (1 - rollOffFactor * (distance - referenceDistance) / (maxDistance - referenceDistance));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

double VolumeCulling_PS::exponentClamped(double distance, ALfloat rollOffFactor,
  ALfloat referenceDistance, ALfloat maxDistance) {
  return
    std::pow((distance / referenceDistance), -1 * rollOffFactor);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool VolumeCulling_PS::requiresUpdate() const {
  return false;  
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void VolumeCulling_PS::update() {
    
}

} // namespace cs::audio