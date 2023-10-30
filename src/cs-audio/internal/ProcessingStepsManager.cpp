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
# include "../processingSteps/Spatialization_PS.hpp"

namespace cs::audio {

std::shared_ptr<ProcessingStepsManager> ProcessingStepsManager::createProcessingStepsManager() {
  static auto psManager = std::shared_ptr<ProcessingStepsManager>(new ProcessingStepsManager());
  return psManager;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

ProcessingStepsManager::ProcessingStepsManager() 
  : mPipelines(std::map<AudioController*, std::set<std::shared_ptr<ProcessingStep>>>()) {  
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void ProcessingStepsManager::createPipeline(std::vector<std::string> processingSteps, 
  AudioController* audioController) {
  
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
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void ProcessingStepsManager::createPipeline(AudioController* audioController) {
  std::set<std::shared_ptr<ProcessingStep>> pipeline;
  pipeline.insert(Default_PS::create());
  mPipelines[audioController] = pipeline;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::shared_ptr<ProcessingStep> ProcessingStepsManager::getProcessingStep(std::string processingStep) {

  if (processingStep == "Spatialization") {
    return Spatialization_PS::create();
  }

  // ...

  logger().warn("Audio Processing Warning: Processing step '{}' is not defined!", processingStep);
  return nullptr;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::shared_ptr<std::vector<std::string>> ProcessingStepsManager::process(ALuint openAlId, 
  AudioController* audioController, std::shared_ptr<std::map<std::string, std::any>> settings) {

  auto failedSettings = std::make_shared<std::vector<std::string>>();
  for (auto step : mPipelines[audioController]) {
    step->process(openAlId, settings, failedSettings);
  }
  return failedSettings;
}

void ProcessingStepsManager::callPsUpdateFunctions() {
  for (auto psPtr : mUpdateProcessingSteps) {
    psPtr->update();
  }
}

} // namespace cs::audio