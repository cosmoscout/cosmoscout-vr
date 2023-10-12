////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "AudioController.hpp"
#include "internal/BufferManager.hpp"
#include "internal/ProcessingStepsManager.hpp"
#include "Source.hpp"
#include "SourceGroup.hpp"

namespace cs::audio {

AudioController::AudioController(
  std::shared_ptr<BufferManager> bufferManager, 
  std::shared_ptr<ProcessingStepsManager> processingStepsManager,
  std::vector<std::string> processingStpes,
  int audioControllerId) 
  : mBufferManager(std::move(bufferManager))
  , mProcessingStepsManager(std::move(processingStepsManager)) 
  , mGlobalPluginSettings(std::make_shared<std::map<std::string, std::any>>())
  , mAudioControllerId(std::move(audioControllerId)) {

  // TODO:
  // mProcessingStepsManager->createPipeline(processingStpes);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::shared_ptr<audio::SourceGroup> AudioController::createSourceGroup() {
  return std::make_shared<audio::SourceGroup>(mProcessingStepsManager);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::shared_ptr<audio::Source> AudioController::createSource(std::string file) {
  return std::make_shared<audio::Source>(mBufferManager, mProcessingStepsManager, file);
} 

////////////////////////////////////////////////////////////////////////////////////////////////////

void AudioController::set(std::string key, std::any value) {
    mGlobalPluginSettings->operator[](key) = value;
}

} // namespace cs::audio