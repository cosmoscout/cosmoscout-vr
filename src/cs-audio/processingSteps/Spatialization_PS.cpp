////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "Spatialization_PS.hpp"
#include "../internal/alErrorHandling.hpp"
#include "../logger.hpp"

#include <AL/al.h>
#include <glm/fwd.hpp>
#include <memory>
#include <vector>
#include <string>

namespace cs::audio {

std::shared_ptr<ProcessingStep> Spatialization_PS::create() {
  static auto spatialization_ps = std::shared_ptr<Spatialization_PS>(new Spatialization_PS());
  return spatialization_ps;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

Spatialization_PS::Spatialization_PS() {}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Spatialization_PS::process(ALuint openAlId, 
  std::shared_ptr<std::map<std::string, std::any>> settings,
  std::shared_ptr<std::vector<std::string>> failedSettings) {
  
  if (settings->find("position") != settings->end()) { 
  
    if (!validatePosition(settings->at("position"))) {
      failedSettings->push_back("position");
      return;
    }

    glm::dvec3 pos = std::any_cast<glm::dvec3>(settings->at("position"));
    alSource3f(openAlId, AL_POSITION, 
      (ALfloat)pos.x, 
      (ALfloat)pos.y, 
      (ALfloat)pos.z);
    
    if (alErrorHandling::errorOccurred()) {
      logger().warn("Failed to set source position!");
    }
    // add source to updateList
    //mUpdateList[openAlId] = std::any_cast<audioTypes::Position>(settings->at("position"));
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool Spatialization_PS::validatePosition(std::any position) {
  glm::dvec3 positionValue;
  
  try {
    positionValue = std::any_cast<glm::dvec3>(position);
  } catch (const std::bad_any_cast&) {
    logger().warn("Audio source settings error! Wrong type used for center setting! Allowed Type: glm::dvec3");
    return false;
  }

  return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool Spatialization_PS::requiresUpdate() const {
  return false;  
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Spatialization_PS::update() {
  /*
  for (auto source : mUpdateList) {
    
    auto celesObj = mSolarSystem->getObject(source.second.center);
    if (celesObj == nullptr) { continue; }

    glm::dvec3 sourcePos = celesObj->getObserverRelativePosition(source.second.coordinates);
    sourcePos *= static_cast<float>(mSolarSystem->getObserver().getScale());

    alSource3f(source.first, AL_POSITION, 
      (ALfloat)sourcePos.x, 
      (ALfloat)sourcePos.y, 
      (ALfloat)sourcePos.z);
  }
  */
}

} // namespace cs::audio