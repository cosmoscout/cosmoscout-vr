////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CS_AUDIO_PS_SPATIALIZATION_UTILS_HPP
#define CS_AUDIO_PS_SPATIALIZATION_UTILS_HPP

#include "cs_audio_export.hpp"
#include "AL/al.h"
#include <glm/fwd.hpp>
#include <glm/detail/type_vec3.hpp>
#include <chrono>
#include "../internal/SourceBase.hpp"

namespace cs::audio {

class CS_AUDIO_EXPORT SpatializationUtils {
 public:
  SpatializationUtils();
  
  /// @brief Calculates and applies the velocity for each spatialized source via the change of position
  void calculateVelocity();

  /// @brief Rotates the the position of source around the vista viewer orientation. This is needed
  /// because the head rotation is not considered in CelestialObject::getObserverRelativePosition()
  /// when using a VR headset.
  /// @param position Relative position to observer
  void rotateSourcePosByViewer(glm::dvec3& position);

  /// @brief Sets the position and Velocity of a source to zero and removes said source from the 
  /// update list.
  /// @param openAlId id of source to reset
  /// @return True if successful. False otherwise.
  bool resetSpatialization(ALuint openAlId);
 
 protected:
  /// Struct to hold all necessary information regarding a spatialized source
  struct SourceContainer {
    std::weak_ptr<SourceBase> sourcePtr;
    glm::dvec3 currentPos;
    glm::dvec3 lastPos;
  };

  /// List of all Source which have a position
  std::map<ALuint, SourceContainer> mSourcePositions;
  /// Point in time since the last calculateVelocity() call
  std::chrono::system_clock::time_point mLastTime;
};

} // namespace cs::audio

#endif // CS_AUDIO_PS_SPATIALIZATION_UTILS_HPP
