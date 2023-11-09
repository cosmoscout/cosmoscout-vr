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
class ProcessingStep;

class CS_AUDIO_EXPORT ProcessingStepsManager {
 public:
  ProcessingStepsManager(const ProcessingStepsManager& obj) = delete;
  ProcessingStepsManager(ProcessingStepsManager&&) = delete;

  ProcessingStepsManager& operator=(const ProcessingStepsManager&) = delete;
  ProcessingStepsManager& operator=(ProcessingStepsManager&&) = delete;

  static std::shared_ptr<ProcessingStepsManager> createProcessingStepsManager(std::shared_ptr<core::Settings> settings);

  /// @brief creates a new Pipeline for an AudioController
  /// @param processingSteps List of name of all processing steps, which should be part of the pipeline
  /// @param audioController Pointer to audioController requesting the pipeline
  void createPipeline(std::vector<std::string> processingSteps, std::shared_ptr<AudioController> audioController);

  /// @brief Calls all processing steps part of the audioControllers pipeline for a source and applies all provided settings.
  /// @param source Source to process.
  /// @param audioController AudioController on which the source lives. Specifies the pipeline.
  /// @param sourceSettings Settings to apply to the provided source
  /// @return List of settings keys that failed when trying to apply the settings to the source.
  std::shared_ptr<std::vector<std::string>> process(
    std::shared_ptr<Source> source, 
    std::shared_ptr<AudioController> audioController,
    std::shared_ptr<std::map<std::string, std::any>> sourceSettings);

  /// @brief This functions will call all update functions of processing steps that are active and require
  /// an every frame update.
  void callPsUpdateFunctions();

 private:                                                                            
  /// Holds all pipelines and their corresponding audioController                                                                       
  std::map<std::shared_ptr<AudioController>, std::set<std::shared_ptr<ProcessingStep>>> mPipelines;
  /// List that contains all processing steps that require an update call every frame
  std::set<std::shared_ptr<ProcessingStep>>                             mUpdateProcessingSteps;
  std::shared_ptr<core::Settings> mSettings;

  ProcessingStepsManager(std::shared_ptr<core::Settings> settings);

  /// @brief Searches for and creates a processing step when defining a pipeline. If you want to add a new 
  /// processing step then you need to define the name and the corresponding create call here.
  /// @param processingStep Name of the processing step to create
  /// @return Pointer to the processing step instance. Nullptr if processing step was not found. 
  std::shared_ptr<ProcessingStep> getProcessingStep(std::string processingStep);
  
  /// @brief Check if any processing step was removed during a redefinition of a pipeline that
  /// is part of mUpdateProcessingSteps. If so, removes the given processing step from mUpdateProcessingSteps.
  void removeObsoletePsFromUpdateList();
};

} // namespace cs::audio

#endif // CS_AUDIO_PROCESSING_STEPS_MANAGER_HPP
