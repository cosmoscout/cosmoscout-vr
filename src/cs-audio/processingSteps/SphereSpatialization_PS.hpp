////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CS_AUDIO_PS_SPHERE_SOURCE_HPP
#define CS_AUDIO_PS_SPHERE_SOURCE_HPP

#include "cs_audio_export.hpp"
#include "ProcessingStep.hpp"
#include "../internal/SourceBase.hpp"
#include <memory>
#include <AL/al.h>
#include <glm/fwd.hpp>
#include <chrono>

namespace cs::audio {

class CS_AUDIO_EXPORT ScaledSphereSpatialization_PS : public ProcessingStep {
 public:

  static std::shared_ptr<ProcessingStep> create();

  void process(std::shared_ptr<SourceBase> source, 
   std::shared_ptr<std::map<std::string, std::any>> settings,
   std::shared_ptr<std::vector<std::string>> failedSettings);

  bool processScaling(std::shared_ptr<SourceBase> source, std::any value, std::any obsScale);

  bool requiresUpdate() const;

  void update();

 private:
  /// Struct to hold all necessary information regarding a spatialized source
  struct SourceContainer {
    std::weak_ptr<SourceBase> sourcePtr;
    glm::dvec3 currentPos;
    glm::dvec3 lastPos;
  };

  ScaledSphereSpatialization_PS();
  double getSourceScale(glm::dvec3 sourcePosToObserver, double obsScale);
  void calculateVelocity();

  /// List of all Source which have a position
  std::map<ALuint, SourceContainer> mSourcePositions;
  /// Point in time since the last calculateVelocity() call
  std::chrono::system_clock::time_point mLastTime;
};

} // namespace cs::audio

#endif // CS_AUDIO_PS_SPHERE_SOURCE_HPP
