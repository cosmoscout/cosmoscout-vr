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

////////////////////////////////////////////////////////////////////////////////////////////////////

ProcessingStepsManager::ProcessingStepsManager() 
  : mPipelines(std::map<std::shared_ptr<AudioController>, std::set<std::shared_ptr<ProcessingStep>>>())
  , mExistingProcessingSteps(std::map<std::string, std::shared_ptr<ProcessingStep>>()) {
  
  mExistingProcessingSteps["Default"] = std::make_shared<Default_PS>(); 
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void ProcessingStepsManager::createPipeline(std::vector<std::string> processingSteps, 
  std::shared_ptr<AudioController> audioController) {
  
  std::set<std::shared_ptr<ProcessingStep>> pipeline;
  pipeline.insert(mExistingProcessingSteps["Default"]);

  for (std::string processingStep : processingSteps) {
    auto ps = getProcessingStep(processingStep);
    if (ps != nullptr) {
      pipeline.insert(ps);
    }
  }

  mPipelines[audioController] = pipeline;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::shared_ptr<ProcessingStep> ProcessingStepsManager::getProcessingStep(std::string processingStep) {
  // Search for processing step and reuse it if it already exists:
  if (auto search = mExistingProcessingSteps.find(processingStep); search != mExistingProcessingSteps.end()) {
    return mExistingProcessingSteps[processingStep];
  }

  // Create not yet existing processing step:
  if (processingStep == "Spatialization") {
    mExistingProcessingSteps[processingStep] = std::make_shared<Spatialization_PS>();
    return mExistingProcessingSteps[processingStep];
  }

  // ...

  logger().warn("Audio Processing Warning: Unable to create '{}' processing step!", processingStep);
  return nullptr;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void ProcessingStepsManager::process(ALuint openAlId, std::shared_ptr<AudioController> audioController, 
  std::shared_ptr<std::map<std::string, std::any>> settings) {

  for (auto step : mPipelines[audioController]) {
    step->process(openAlId, settings);
  }
}

} // namespace cs::audio