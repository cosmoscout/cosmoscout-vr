////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "SpatializationUtils.hpp"
#include "../internal/AlErrorHandling.hpp"
#include "../logger.hpp"
#include <VistaKernel/DisplayManager/VistaDisplayManager.h>
#include <VistaKernel/DisplayManager/VistaDisplaySystem.h>
#include <VistaKernel/VistaSystem.h>
#include <glm/fwd.hpp>

namespace cs::audio {

SpatializationUtils::SpatializationUtils(bool stationaryOutputDevice)
    : mSourcePositions(std::map<ALuint, SourceContainer>())
    , mLastTime(std::chrono::system_clock::now())
    , mStationaryOutputDevice(stationaryOutputDevice) {
}

void SpatializationUtils::calculateVelocity() {
  std::chrono::system_clock::time_point currentTime      = std::chrono::system_clock::now();
  std::chrono::duration<float>          elapsed_seconds  = currentTime - mLastTime;
  auto                                  elapsed_secondsf = elapsed_seconds.count();

  for (auto source : mSourcePositions) {

    if (source.second.sourcePtr.expired()) {
      mSourcePositions.erase(source.first);
      continue;
    }

    glm::dvec3 velocity;
    ALuint     openAlId = source.second.sourcePtr.lock()->getOpenAlId();

    if (source.second.currentPos != source.second.lastPos) {
      glm::dvec3 posDelta                = source.second.currentPos - source.second.lastPos;
      velocity.x                         = posDelta.x / elapsed_secondsf;
      velocity.y                         = posDelta.y / elapsed_secondsf;
      velocity.z                         = posDelta.z / elapsed_secondsf;
      mSourcePositions[openAlId].lastPos = source.second.currentPos;

    } else {
      velocity.x = 0;
      velocity.y = 0;
      velocity.z = 0;
    }

    alSource3f(
        openAlId, AL_VELOCITY, (ALfloat)velocity.x, (ALfloat)velocity.y, (ALfloat)velocity.z);

    if (AlErrorHandling::errorOccurred()) {
      logger().warn("Failed to set source velocity!");
    }
  }

  mLastTime = currentTime;
}

void SpatializationUtils::compensateSpeakerRotation(glm::dvec3& position) {
  auto viewerOrient = GetVistaSystem()
                          ->GetDisplayManager()
                          ->GetDisplaySystem()
                          ->GetDisplaySystemProperties()
                          ->GetViewerOrientation();
  viewerOrient.Invert();
  VistaVector3D vista3d((float)position.x, (float)position.y, (float)position.z);
  VistaVector3D sourceRelPosToObsRot = viewerOrient.Rotate(vista3d);
  position.x                         = sourceRelPosToObsRot[0];
  position.y                         = sourceRelPosToObsRot[1];
  position.z                         = sourceRelPosToObsRot[2];
}

bool SpatializationUtils::resetSpatialization(ALuint openAlId) {
  // TODO: erase source from list
  mSourcePositions.erase(openAlId);

  alSourcei(openAlId, AL_SOURCE_RELATIVE, AL_TRUE);
  if (AlErrorHandling::errorOccurred()) {
    logger().warn("Failed to reset source position specification to relative!");
    return false;
  }

  alSource3f(openAlId, AL_POSITION, (ALfloat)0.f, (ALfloat)0.f, (ALfloat)0.f);
  if (AlErrorHandling::errorOccurred()) {
    logger().warn("Failed to reset source position!");
    return false;
  }

  alSource3f(openAlId, AL_VELOCITY, (ALfloat)0.f, (ALfloat)0.f, (ALfloat)0.f);
  if (AlErrorHandling::errorOccurred()) {
    logger().warn("Failed to reset source velocity!");
    return false;
  }
  return true;
}

} // namespace cs::audio