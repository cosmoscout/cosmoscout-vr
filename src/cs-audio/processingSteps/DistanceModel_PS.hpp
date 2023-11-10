////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CS_AUDIO_PS_DISTANCE_MODEL_HPP
#define CS_AUDIO_PS_DISTANCE_MODEL_HPP

#include "cs_audio_export.hpp"
#include "ProcessingStep.hpp"
#include "../internal/SourceBase.hpp"
#include "../../cs-core/Settings.hpp"
#include <memory>
#include <AL/al.h>
#include <glm/fwd.hpp>
#include <chrono>

namespace cs::audio {

class CS_AUDIO_EXPORT DistanceModel_PS : public ProcessingStep {
 public:

  static std::shared_ptr<ProcessingStep> create();

  void process(std::shared_ptr<SourceBase> source, 
   std::shared_ptr<std::map<std::string, std::any>> settings,
   std::shared_ptr<std::vector<std::string>> failedSettings);

  bool requiresUpdate() const;

  void update();

 private:
  DistanceModel_PS();

  bool processModel(ALuint openALId, std::any model);
  bool processFallOffStart(ALuint openALId, std::any fallOffStart);
  bool processFallOffEnd(ALuint openALId, std::any fallOffEnd);
  bool processFallOffFactor(ALuint openALId, std::any fallOffFactor);
};

} // namespace cs::audio

#endif // CS_AUDIO_PS_DISTANCE_MODEL_HPP