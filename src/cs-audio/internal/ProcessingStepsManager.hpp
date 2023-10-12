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

#include <AL/al.h>
#include <map>
#include <set>

namespace cs::audio {

class CS_AUDIO_EXPORT ProcessingStepsManager {
 public:
  ProcessingStepsManager(const ProcessingStepsManager& obj) = delete;
  ProcessingStepsManager(ProcessingStepsManager&&) = delete;

  ProcessingStepsManager& operator=(const ProcessingStepsManager&) = delete;
  ProcessingStepsManager& operator=(ProcessingStepsManager&&) = delete;

  ProcessingStepsManager();

  void createPipeline(std::vector<std::string> processingSteps, int audioControllerId);
  void process(ALuint openAlId, int audioControllerId,
    std::shared_ptr<std::map<std::string, std::any>> sourceSettings);
    
 private:                                                                                                                                                     
  std::map<int, std::set<std::shared_ptr<ProcessingStep>>> mPipelines;
  std::set<std::shared_ptr<ProcessingStep>> existingProcessingSteps;
  ProcessingStep createProcessingStep(std::string processingStep);
};

} // namespace cs::audio

#endif // CS_AUDIO_PROCESSING_STEPS_MANAGER_HPP
