////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "AudioController.hpp"
#include "internal/BufferManager.hpp"
#include "internal/ProcessingStepsManager.hpp"
#include "internal/SettingsMixer.hpp"
#include "internal/UpdateInstructor.hpp"
#include "Source.hpp"
#include "StreamingSource.hpp"
#include "SourceGroup.hpp"

namespace cs::audio {

AudioController::AudioController(
  std::shared_ptr<BufferManager> bufferManager, 
  std::shared_ptr<ProcessingStepsManager> processingStepsManager,
  std::shared_ptr<UpdateConstructor> updateConstructor) 
  : SourceSettings()
  , std::enable_shared_from_this<AudioController>()
  , mBufferManager(std::move(bufferManager))
  , mProcessingStepsManager(std::move(processingStepsManager))
  , mUpdateInstructor(std::make_shared<UpdateInstructor>())
  , mUpdateConstructor(std::move(updateConstructor)) {
  setUpdateInstructor(mUpdateInstructor);  
}
  removeFromUpdateList();

////////////////////////////////////////////////////////////////////////////////////////////////////

std::shared_ptr<SourceGroup> AudioController::createSourceGroup() {
  auto group = std::make_shared<SourceGroup>(mUpdateInstructor, mUpdateConstructor, shared_from_this());
  mGroups.push_back(group);
  return group;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::shared_ptr<Source> AudioController::createSource(std::string file) {
  auto source = std::make_shared<Source>(mBufferManager, file, mUpdateInstructor);
  mSources.push_back(source);

  // apply audioController settings to newly creates source
  if (!mCurrentSettings->empty()) {
    mUpdateConstructor->applyCurrentControllerSettings(source, shared_from_this(), mCurrentSettings);
  }
  return source;
} 

////////////////////////////////////////////////////////////////////////////////////////////////////

std::shared_ptr<StreamingSource> AudioController::createStreamingSource(std::string file) {
  auto source = std::make_shared<StreamingSource>(file, mUpdateInstructor);
  mSources.push_back(source);
  mStreams.push_back(source);

  // apply audioController settings to newly creates source
  if (!mCurrentSettings->empty()) {
    mUpdateConstructor->applyCurrentControllerSettings(source, shared_from_this(), mCurrentSettings);
  }
  return source;
} 

////////////////////////////////////////////////////////////////////////////////////////////////////

void AudioController::setPipeline(std::vector<std::string> processingSteps) {
  mProcessingStepsManager->createPipeline(processingSteps, shared_from_this());
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void AudioController::update() {
  
  auto updateInstructions = mUpdateInstructor->createUpdateInstruction();

  // updateInstructions.print();

  // update every source and group with plugin settings
  if (updateInstructions.updateAll) {
    mUpdateConstructor->updateAll(
      std::make_shared<std::vector<std::shared_ptr<SourceBase>>>(mSources),
      std::make_shared<std::vector<std::shared_ptr<SourceGroup>>>(mGroups),
      shared_from_this());
    return;
  }

  // update changed groups with member sources
  if (updateInstructions.updateWithGroup->size() > 0) {
    mUpdateConstructor->updateGroups(
      updateInstructions.updateWithGroup,
      std::make_shared<std::vector<std::shared_ptr<SourceGroup>>>(mGroups),
      shared_from_this());
  }

  // update leftover changed sources
  if (updateInstructions.updateSourceOnly->size() > 0) {
    mUpdateConstructor->updateSources(
      updateInstructions.updateSourceOnly,
      shared_from_this());
  }
}

void AudioController::updateStreamingSources() {
  for (auto stream : mStreams) {
    stream->updateStream();
  }
}

std::vector<std::shared_ptr<SourceBase>> AudioController::getSources() const {
  return std::vector<std::shared_ptr<SourceBase>>(mSources);
}

void AudioController::addToUpdateList() {
  mUpdateInstructor->update(shared_from_this());
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void AudioController::removeFromUpdateList() {
  mUpdateInstructor->removeUpdate(shared_from_this());
}

} // namespace cs::audio