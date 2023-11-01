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
#include "../AudioController.hpp"

#include <AL/al.h>
#include <map>
#include <set>

namespace cs::audio {

class AudioController;

class CS_AUDIO_EXPORT ProcessingStepsManager {
 public:
  ProcessingStepsManager(const ProcessingStepsManager& obj) = delete;
  ProcessingStepsManager(ProcessingStepsManager&&) = delete;

  ProcessingStepsManager& operator=(const ProcessingStepsManager&) = delete;
  ProcessingStepsManager& operator=(ProcessingStepsManager&&) = delete;

  static std::shared_ptr<ProcessingStepsManager> createProcessingStepsManager();

  void createPipeline(std::vector<std::string> processingSteps, AudioController* audioController);
  void createPipeline(AudioController* audioController);
  std::shared_ptr<std::vector<std::string>> process(std::shared_ptr<Source> source, AudioController* audioController,
    std::shared_ptr<std::map<std::string, std::any>> sourceSettings);

  /// @brief This functions will call all update function of processing steps that are active and required.
  void callPsUpdateFunctions();

 private:                                                                          
  // TODO: replace AudioController* with smart pointer                                                                           
  std::map<AudioController*, std::set<std::shared_ptr<ProcessingStep>>> mPipelines;
  /// Contains all processing steps that require an update call every frame
  std::set<std::shared_ptr<ProcessingStep>>                             mUpdateProcessingSteps;

  ProcessingStepsManager();
  std::shared_ptr<ProcessingStep> getProcessingStep(std::string processingStep);
  
  /// Check if any PS was removed form a pipeline that is set in mUpdateProcessingSteps and
  /// if so, remove the given PS from mUpdateProcessingSteps.
  void removeObsoletePsFromUpdateList();
};

} // namespace cs::audio

#endif // CS_AUDIO_PROCESSING_STEPS_MANAGER_HPP
