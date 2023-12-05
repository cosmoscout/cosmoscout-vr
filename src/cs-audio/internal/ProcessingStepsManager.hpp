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

/// @brief This class manages the creation, deletion and calling of all processing steps.
/// This class should only be instantiated once.
class CS_AUDIO_EXPORT ProcessingStepsManager {
 public:
  ProcessingStepsManager(const ProcessingStepsManager& obj) = delete;
  ProcessingStepsManager(ProcessingStepsManager&&) = delete;

  ProcessingStepsManager& operator=(const ProcessingStepsManager&) = delete;
  ProcessingStepsManager& operator=(ProcessingStepsManager&&) = delete;

  static std::shared_ptr<ProcessingStepsManager> createProcessingStepsManager(std::shared_ptr<core::Settings> settings);
  ~ProcessingStepsManager();
  
  /// @brief Creates a new Pipeline. A pipeline is a just a list of processing steps that should 
  /// be active for all sources of an audio controller. 
  /// @param processingSteps List of processing step names, which should be part of the pipeline
  /// @param audioControllerId ID of the audioController requesting the pipeline
  void createPipeline(std::vector<std::string> processingSteps, int audioControllerId);

  /// @brief Deletes a pipeline completely. Gets called during the deconstruction
  /// of an audio controller. 
  void removeAudioController(int audioControllerId);

  /// @brief Calls all processing steps part of the audioControllers pipeline for a source and 
  /// tries to apply all provided settings.
  /// @param source Source to process.
  /// @param audioControllerId AudioController on which the source lives. Specifies the pipeline.
  /// @param sourceSettings Settings to apply to the provided source
  /// @return List of settings that failed when trying to apply the settings to the source.
  std::shared_ptr<std::vector<std::string>> process(
    std::shared_ptr<SourceBase> source, 
    int audioControllerId,
    std::shared_ptr<std::map<std::string, std::any>> sourceSettings);

  /// @brief This functions will call all update functions of processing steps that are 
  /// active and require an update every frame.
  void callPsUpdateFunctions();

  ProcessingStepsManager(std::shared_ptr<core::Settings> settings);
 private:                                                                            
  /// Holds all pipelines and their corresponding audioController                                                                       
  std::map<int, std::set<std::shared_ptr<ProcessingStep>>> mPipelines;
  /// List of processing steps that require an update call every frame
  std::set<std::shared_ptr<ProcessingStep>>                mUpdateProcessingSteps;
  std::shared_ptr<core::Settings>                          mSettings;

  /// @brief Searches for and creates a processing step when defining a pipeline. If you want to add 
  /// a new processing step then you need to define the name and the corresponding create call here.
  /// @param processingStep Name of the processing step to create
  /// @return Pointer to the processing step instance. Nullptr if processing step was not found. 
  std::shared_ptr<ProcessingStep> getProcessingStep(std::string processingStep);
  
  /// @brief Check if any processing step was removed during a redefinition of a pipeline that
  /// is part of mUpdateProcessingSteps. If so, removes the given processing step from mUpdateProcessingSteps.
  void removeObsoletePsFromUpdateList();
};

} // namespace cs::audio

#endif // CS_AUDIO_PROCESSING_STEPS_MANAGER_HPP
