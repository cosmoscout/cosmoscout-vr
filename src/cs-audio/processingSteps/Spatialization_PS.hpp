////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CS_AUDIO_PS_SPATIALIZATION_HPP
#define CS_AUDIO_PS_SPATIALIZATION_HPP

#include "cs_audio_export.hpp"
#include "ProcessingStep.hpp"
#include "../Source.hpp"

#include <AL/al.h>
#include <glm/fwd.hpp>
#include <chrono>

namespace cs::audio {

class CS_AUDIO_EXPORT Spatialization_PS : public ProcessingStep {
 public:

  static std::shared_ptr<ProcessingStep> create();

  void process(std::shared_ptr<Source> source, 
    std::shared_ptr<std::map<std::string, std::any>> settings,
    std::shared_ptr<std::vector<std::string>> failedSettings) override;

  bool requiresUpdate() const;

  void update();

 private:

  struct SourceContainer {
    std::weak_ptr<Source> sourcePtr;
    glm::dvec3 currentPos;
    glm::dvec3 lastPos;
  };

  Spatialization_PS();

  bool processPosition(std::shared_ptr<Source> source, std::any position);
  void calculateVelocity();
  std::map<ALuint, SourceContainer> mSourcePositions;
  std::chrono::system_clock::time_point mLastTime;
};

} // namespace cs::audio

#endif // CS_AUDIO_PS_SPATIALIZATION_HPP
