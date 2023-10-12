////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "ProcessingStepsManager.hpp"
#include "../../cs-core/Settings.hpp"
#include "../../cs-core/Settings.hpp"
#include <set>

// processingSteps:
# include "../processingSteps/Default_PS.hpp"
# include "../processingSteps/Spatialization_PS.hpp"

namespace cs::audio {

////////////////////////////////////////////////////////////////////////////////////////////////////

ProcessingStepsManager::ProcessingStepsManager() {
  activeProcessingSteps.push_back(std::make_shared<Default_PS>()); 

  // setProcessingSteps();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void ProcessingStepsManager::createPipeline(std::vector<std::string> processingSteps, int audioControllerId) {
  std::set<std::shared_ptr<ProcessingStep>> pipeline;

  for (std::string processingStep : processingSteps) {

    if (processingStep == "Spatialization") {
      pipeline.insert(std::make_shared<Spatialization_PS>());
      continue;
    }

    // ...
  }

  mPipelines[audioControllerId] = pipeline;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

ProcessingStep ProcessingStepsManager::createProcessingStep(std::string processingStep) {
  
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void ProcessingStepsManager::process(ALuint openAlId, int audioControllerId, std::shared_ptr<std::map<std::string, std::any>> settings) {
  for (auto step : activeProcessingSteps) {
    step->process(openAlId, settings);
  }
}

} // namespace cs::audio