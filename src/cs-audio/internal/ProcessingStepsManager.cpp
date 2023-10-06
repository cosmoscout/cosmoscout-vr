////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "ProcessingStepsManager.hpp"
#include "../../cs-core/Settings.hpp"
#include "../../cs-core/Settings.hpp"

// processingSteps:
# include "../processingSteps/Default_PS.hpp"
# include "../processingSteps/Spatialization_PS.hpp"

namespace cs::audio {

////////////////////////////////////////////////////////////////////////////////////////////////////

ProcessingStepsManager::ProcessingStepsManager(std::shared_ptr<core::Settings> settings) {
  activeProcessingSteps.push_back(std::make_shared<Default_PS>()); 

  // setProcessingSteps();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void ProcessingStepsManager::setProcessingSteps(std::vector<std::string> processingSteps) {
  for (std::string processingStep : processingSteps) {

    if (processingStep == "Spatialization") {
      activeProcessingSteps.push_back(std::make_shared<Spatialization_PS>());
      continue;
    }

    // ...
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void ProcessingStepsManager::process(ALuint openAlId, std::shared_ptr<SourceSettings> settings) {
  for (auto step : activeProcessingSteps) {
    step->process(openAlId, settings);
  }
}

} // namespace cs::audio