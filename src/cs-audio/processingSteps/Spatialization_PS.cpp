////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "Spatialization_PS.hpp"
#include "../internal/alErrorHandling.hpp"
#include "../logger.hpp"

#include <AL/al.h>
#include <glm/detail/type_vec3.hpp>
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
  
  if (auto search = settings->find("position"); search != settings->end()) { 
    if (!processPosition(openAlId, search->second)) {
      failedSettings->push_back("position");
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool Spatialization_PS::processPosition(ALuint openAlId, std::any value) {
  glm::dvec3 positionValue;
  
  if (value.type() != typeid(glm::dvec3)) {
    logger().warn("Audio source settings error! Wrong type used for position setting! Allowed Type: glm::dvec3");
    return false;
  }

  alSourcei(openAlId, AL_SOURCE_RELATIVE, AL_FALSE);
  if (alErrorHandling::errorOccurred()) {
    logger().warn("Failed to set source position specification to absolute!");
    return false;
  }

  positionValue = std::any_cast<glm::dvec3>(value);

  alSource3f(openAlId, AL_POSITION, 
    (ALfloat)positionValue.x, 
    (ALfloat)positionValue.y, 
    (ALfloat)positionValue.z);
  
  if (alErrorHandling::errorOccurred()) {
    logger().warn("Failed to set source position!");
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

}

} // namespace cs::audio