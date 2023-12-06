////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "PointSpatialization_PS.hpp"
#include "../internal/AlErrorHandling.hpp"
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

PointSpatialization_PS::PointSpatialization_PS() {}

////////////////////////////////////////////////////////////////////////////////////////////////////

void PointSpatialization_PS::process(std::shared_ptr<SourceBase> source, 
  std::shared_ptr<std::map<std::string, std::any>> settings,
  std::shared_ptr<std::vector<std::string>> failedSettings) {
  
  if (auto search = settings->find("position"); search != settings->end()) { 
    if (!processPosition(source, search->second)) {
      failedSettings->push_back("position");
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool PointSpatialization_PS::processPosition(std::shared_ptr<SourceBase> source, std::any value) {
  
  if (value.type() != typeid(glm::dvec3)) {

    // remove position
    if (value.type() == typeid(std::string) && std::any_cast<std::string>(value) == "remove") { 
      return resetSpatialization(source->getOpenAlId());
    }

    logger().warn("Audio source settings error! Wrong type used for position setting! Allowed Type: glm::dvec3");
    return false;
  }

  ALuint openAlId = source->getOpenAlId();

  alSourcei(openAlId, AL_SOURCE_RELATIVE, AL_FALSE);
  if (AlErrorHandling::errorOccurred()) {
    logger().warn("Failed to set source position specification to absolute!");
    return false;
  }

  glm::dvec3 positionValue = std::any_cast<glm::dvec3>(value);
  rotateSourcePosByViewer(positionValue);

  alSource3f(openAlId, AL_POSITION, 
    (ALfloat)positionValue.x, 
    (ALfloat)positionValue.y, 
    (ALfloat)positionValue.z);
  
  if (AlErrorHandling::errorOccurred()) {
    logger().warn("Failed to set source position!");
    return false;
  }

  mSourcePositions[openAlId] = SourceContainer{std::weak_ptr<SourceBase>(source), positionValue, positionValue};
  return true;
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