////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CS_AUDIO_PROCESSING_STEPS_MANAGER_HPP
#define CS_AUDIO_PROCESSING_STEPS_MANAGER_HPP

#include "cs_audio_export.hpp"
#include "../../cs-core/Settings.hpp"
#include "../processingSteps/ProcessingStep.hpp"
#include "../SourceSettings.hpp"

#include <AL/al.h>

namespace cs::audio {

class CS_AUDIO_EXPORT ProcessingStepsManager {
 public:
  ProcessingStepsManager(const ProcessingStepsManager& obj) = delete;
  ProcessingStepsManager(ProcessingStepsManager&&) = delete;

  ProcessingStepsManager& operator=(const ProcessingStepsManager&) = delete;
  ProcessingStepsManager& operator=(ProcessingStepsManager&&) = delete;

  ProcessingStepsManager(std::shared_ptr<core::Settings> settings);
  void process(ALuint openAlId, std::shared_ptr<SourceSettings> settings);

 private:
  std::vector<std::shared_ptr<ProcessingStep>> activeProcessingSteps;
  
  void setProcessingSteps(std::vector<std::string> processingSteps);
};

} // namespace cs::audio

#endif // CS_AUDIO_PROCESSING_STEPS_MANAGER_HPP
