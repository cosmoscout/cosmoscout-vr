////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "DistanceModel_PS.hpp"
#include "../../cs-scene/CelestialAnchor.hpp"
#include "../../cs-scene/CelestialSurface.hpp"
#include "../../cs-utils/convert.hpp"
#include "../internal/AlErrorHandling.hpp"
#include "../logger.hpp"
#include <cmath>
#include <glm/gtx/matrix_decompose.hpp>

namespace cs::audio {

std::shared_ptr<ProcessingStep> DistanceModel_PS::create() {
  static auto distanceModel_PS = std::shared_ptr<DistanceModel_PS>(new DistanceModel_PS());
  return distanceModel_PS;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

DistanceModel_PS::DistanceModel_PS() {
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void DistanceModel_PS::process(std::shared_ptr<SourceBase> source,
    std::shared_ptr<std::map<std::string, std::any>>       settings,
    std::shared_ptr<std::vector<std::string>>              failedSettings) {

  ALuint openALId    = source->getOpenAlId();
  auto   searchModel = settings->find("distanceModel");
  if (searchModel != settings->end()) {
    if (!processModel(openALId, searchModel->second)) {
      failedSettings->push_back("distanceModel");
    }
  }

  auto searchFallOffStart = settings->find("fallOffStart");
  if (searchFallOffStart != settings->end()) {
    if (!processFallOffStart(openALId, searchFallOffStart->second)) {
      failedSettings->push_back("fallOffStart");
    }
  }

  auto searchFallOffEnd = settings->find("fallOffEnd");
  if (searchFallOffEnd != settings->end()) {
    if (!processFallOffEnd(openALId, searchFallOffEnd->second)) {
      failedSettings->push_back("fallOffEnd");
    }
  }

  auto searchFallOffFactor = settings->find("fallOffFactor");
  if (searchFallOffFactor != settings->end()) {
    if (!processFallOffFactor(openALId, searchFallOffFactor->second)) {
      failedSettings->push_back("fallOffFactor");
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool DistanceModel_PS::processModel(ALuint openALId, std::any model) {
  if (model.type() != typeid(std::string)) {
    logger().warn("Audio source settings error! Wrong type used for distanceModel setting! Allowed "
                  "Type: std::string");
    return false;
  }
  auto modelValue = std::any_cast<std::string>(model);

  if (modelValue == "remove") {
    modelValue = "inverse";
  }

  if (modelValue == "inverse") {
    alSourcei(openALId, AL_DISTANCE_MODEL, AL_INVERSE_DISTANCE_CLAMPED);
    if (AlErrorHandling::errorOccurred()) {
      logger().warn("Failed to set the distance model of a source!");
      return false;
    }
    return true;
  }

  if (modelValue == "exponent") {
    alSourcei(openALId, AL_DISTANCE_MODEL, AL_EXPONENT_DISTANCE_CLAMPED);
    if (AlErrorHandling::errorOccurred()) {
      logger().warn("Failed to set the distance model of a source!");
      return false;
    }
    return true;
  }

  if (modelValue == "linear") {
    alSourcei(openALId, AL_DISTANCE_MODEL, AL_LINEAR_DISTANCE_CLAMPED);
    if (AlErrorHandling::errorOccurred()) {
      logger().warn("Failed to set the distance model of a source!");
      return false;
    }
    return true;
  }

  if (modelValue == "none") {
    alSourcei(openALId, AL_DISTANCE_MODEL, AL_NONE);
    if (AlErrorHandling::errorOccurred()) {
      logger().warn("Failed to set the distance model of a source!");
      return false;
    }
    return true;
  }

  logger().warn("Audio source settings error! Wrong value passed for distanceModel setting! "
                "Allowed values: 'inverse', 'exponent', 'linear', 'none'");
  return false;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool DistanceModel_PS::processFallOffStart(ALuint openALId, std::any fallOffStart) {
  if (fallOffStart.type() != typeid(float)) {

    if (fallOffStart.type() == typeid(std::string) &&
        std::any_cast<std::string>(fallOffStart) == "remove") {
      alSourcei(openALId, AL_REFERENCE_DISTANCE, 1);
      if (AlErrorHandling::errorOccurred()) {
        logger().warn("Failed to reset the fallOffStart setting of a source!");
        return false;
      }
      return true;
    }

    logger().warn("Audio source settings error! Wrong type used for fallOffStart setting! Allowed "
                  "Type: float");
    return false;
  }

  auto fallOffStartValue = std::any_cast<float>(fallOffStart);

  alSourcef(openALId, AL_REFERENCE_DISTANCE, fallOffStartValue);
  if (AlErrorHandling::errorOccurred()) {
    logger().warn("Failed to set the fallOffStart setting of a source!");
    return false;
  }
  return true;
}

bool DistanceModel_PS::processFallOffEnd(ALuint openALId, std::any fallOffEnd) {
  if (fallOffEnd.type() != typeid(float)) {

    if (fallOffEnd.type() == typeid(std::string) &&
        std::any_cast<std::string>(fallOffEnd) == "remove") {
      alSourcef(openALId, AL_MAX_DISTANCE, static_cast<ALfloat>(std::numeric_limits<float>::max()));
      if (AlErrorHandling::errorOccurred()) {
        logger().warn("Failed to reset the fallOffEnd setting of a source!");
        return false;
      }
      return true;
    }

    logger().warn(
        "Audio source settings error! Wrong type used for fallOffEnd setting! Allowed Type: float");
    return false;
  }

  auto fallOffEndValue = std::any_cast<float>(fallOffEnd);

  alSourcef(openALId, AL_MAX_DISTANCE, fallOffEndValue);
  if (AlErrorHandling::errorOccurred()) {
    logger().warn("Failed to set the fallOffEnd setting of a source!");
    return false;
  }
  return true;
}

bool DistanceModel_PS::processFallOffFactor(ALuint openALId, std::any fallOffFactor) {
  if (fallOffFactor.type() != typeid(float)) {

    if (fallOffFactor.type() == typeid(std::string) &&
        std::any_cast<std::string>(fallOffFactor) == "remove") {
      alSourcei(openALId, AL_ROLLOFF_FACTOR, 1);
      if (AlErrorHandling::errorOccurred()) {
        logger().warn("Failed to reset the fallOffEnd setting of a source!");
        return false;
      }
      return true;
    }

    logger().warn(
        "Audio source settings error! Wrong type used for fallOffEnd setting! Allowed Type: float");
    return false;
  }

  auto fallOffFactorValue = std::any_cast<float>(fallOffFactor);

  alSourcef(openALId, AL_ROLLOFF_FACTOR, fallOffFactorValue);
  if (AlErrorHandling::errorOccurred()) {
    logger().warn("Failed to set the fallOffEnd setting of a source!");
    return false;
  }
  return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool DistanceModel_PS::requiresUpdate() const {
  return false;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void DistanceModel_PS::update() {
}

} // namespace cs::audio