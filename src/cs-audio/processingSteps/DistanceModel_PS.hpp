////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CS_AUDIO_PS_DISTANCE_MODEL_HPP
#define CS_AUDIO_PS_DISTANCE_MODEL_HPP

#include "../../cs-core/Settings.hpp"
#include "../internal/SourceBase.hpp"
#include "ProcessingStep.hpp"
#include "cs_audio_export.hpp"
#include <AL/al.h>
#include <chrono>
#include <glm/fwd.hpp>
#include <memory>

namespace cs::audio {
/*
The DistanceModel_PS introduces distance attenuation controls for a source.
This processing step will only get active if a source has a postion.
---------------------------------------------------------
Name          Type          Range       Description
---------------------------------------------------------
distanceModel std::string   "inverse"   Defines the fallOff Shape.
                            "linear"
                            "exponent"
fallOffStart  float         0.0 -       Distance at which the fallOff Starts. If the distance is
smaller you will hear the source at full volume but still spatialized. fallOffEnd    float 0.0 -
Distance at which the fallOff clamps. The does not disable the source but stops a further fallOff,
meaning the attenuation stays the same beyond this distance. fallOffFactor float         0.0 -
Multiplier to the distance attenuation. If set to 0.0, no attenuation occurs.
---------------------------------------------------------
*/
class CS_AUDIO_EXPORT DistanceModel_PS : public ProcessingStep {
 public:
  static std::shared_ptr<ProcessingStep> create();

  void process(std::shared_ptr<SourceBase>             source,
      std::shared_ptr<std::map<std::string, std::any>> settings,
      std::shared_ptr<std::vector<std::string>>        failedSettings);

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