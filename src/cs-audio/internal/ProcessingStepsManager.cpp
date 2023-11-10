////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "ProcessingStepsManager.hpp"
#include "../logger.hpp"
#include "../AudioController.hpp"

#include <set>

// processingSteps:
# include "../processingSteps/Default_PS.hpp"
# include "../processingSteps/PointSpatialization_PS.hpp"
# include "../processingSteps/DirectPlay_PS.hpp"
# include "../processingSteps/VolumeCulling_PS.hpp"
# include "../processingSteps/DistanceCulling_PS.hpp"

namespace cs::audio {

std::shared_ptr<ProcessingStepsManager> ProcessingStepsManager::createProcessingStepsManager(
  std::shared_ptr<core::Settings> settings) {
  
  static auto psManager = std::shared_ptr<ProcessingStepsManager>(
    new ProcessingStepsManager(std::move(settings)));
  return psManager;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

ProcessingStepsManager::ProcessingStepsManager(std::shared_ptr<core::Settings> settings) 
  : mPipelines(std::map<std::shared_ptr<AudioController>, std::set<std::shared_ptr<ProcessingStep>>>())
  , mSettings(std::move(settings)) {  
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void ProcessingStepsManager::createPipeline(std::vector<std::string> processingSteps, 
  std::shared_ptr<AudioController> audioController) {
  
  std::set<std::shared_ptr<ProcessingStep>> pipeline;
  pipeline.insert(Default_PS::create());

  for (std::string processingStep : processingSteps) {
    auto ps = getProcessingStep(processingStep);
    
    if (ps != nullptr) {
      pipeline.insert(ps);

      if (ps->requiresUpdate()) {
        mUpdateProcessingSteps.insert(ps);
      }
    }
  }

  mPipelines[audioController] = pipeline;
  removeObsoletePsFromUpdateList();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::shared_ptr<ProcessingStep> ProcessingStepsManager::getProcessingStep(std::string processingStep) {

  if (processingStep == "PointSpatialization") {
    return PointSpatialization_PS::create();
  }


  if (processingStep == "DirectPlay") {
    return DirectPlay_PS::create();
  }

  if (processingStep == "VolumeCulling") {
    return VolumeCulling_PS::create(mSettings->mAudio.pVolumeCullingThreshold.get());
  }

  if (processingStep == "DistanceCulling") {
    return DistanceCulling_PS::create(mSettings->mAudio.pDistanceCullingThreshold.get());
  }

  // ...

  logger().warn("Audio Processing Warning: Processing step '{}' is not defined!", processingStep);
  return nullptr;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::shared_ptr<std::vector<std::string>> ProcessingStepsManager::process(
  std::shared_ptr<SourceBase> source, 
  std::shared_ptr<AudioController> audioController, 
  std::shared_ptr<std::map<std::string, std::any>> settings) {

  auto failedSettings = std::make_shared<std::vector<std::string>>();
  for (auto step : mPipelines[audioController]) {
    step->process(source, settings, failedSettings);
  }
  return failedSettings;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void ProcessingStepsManager::callPsUpdateFunctions() {
  for (auto psPtr : mUpdateProcessingSteps) {
    psPtr->update();
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void ProcessingStepsManager::removeObsoletePsFromUpdateList() {
  // get active PS
  std::set<std::shared_ptr<ProcessingStep>> activePS;
  for (auto const& [key, val] : mPipelines) {
    activePS.insert(val.begin(), val.end());
  }

  // get all PS that are in mUpdateProcessingSteps but not in activePS
  std::set<std::shared_ptr<ProcessingStep>> obsoletePS;
  std::set_difference(
    mUpdateProcessingSteps.begin(), mUpdateProcessingSteps.end(),
    activePS.begin(), activePS.end(),
    std::inserter(obsoletePS, obsoletePS.end()));

  // erase obsoletePS from mUpdateProcessingSteps
  for (auto ps : obsoletePS) {
    mUpdateProcessingSteps.erase(ps);
  }
}

} // namespace cs::audio