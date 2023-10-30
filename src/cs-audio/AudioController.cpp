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
  mProcessingStepsManager->createPipeline(this);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::shared_ptr<SourceGroup> AudioController::createSourceGroup() {
  auto group = std::make_shared<SourceGroup>(mUpdateInstructor);
  mGroups.push_back(group);
  return group;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::shared_ptr<Source> AudioController::createSource(std::string file) {
  auto source = std::make_shared<Source>(mBufferManager, mProcessingStepsManager, file, mUpdateInstructor);
  mSources.push_back(source);
  return source;
} 

////////////////////////////////////////////////////////////////////////////////////////////////////

void AudioController::setPipeline(std::vector<std::string> processingSteps) {
  mProcessingStepsManager->createPipeline(processingSteps, this);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void AudioController::update() {
  
  auto updateInstructions = mUpdateInstructor->createUpdateInstruction();

  // updateInstructions.print();

  // update every source and group with plugin settings
  if (updateInstructions.updateAll) {
    mUpdateConstructor->updateAll(
      std::make_shared<std::vector<std::shared_ptr<Source>>>(mSources),
      std::make_shared<std::vector<std::shared_ptr<SourceGroup>>>(mGroups),
      this);
    return;
  }

  // update changed groups with member sources
  if (updateInstructions.updateWithGroup->size() > 0) {
    mUpdateConstructor->updateGroups(
      updateInstructions.updateWithGroup,
      std::make_shared<std::vector<std::shared_ptr<SourceGroup>>>(mGroups),
      this);
  }

  // update leftover changed sources
  if (updateInstructions.updateOnlySource->size() > 0) {
    mUpdateConstructor->updateSources(
      updateInstructions.updateOnlySource,
      this);
  }
}

std::shared_ptr<std::vector<std::shared_ptr<Source>>> AudioController::getSources() const {
  return std::make_shared<std::vector<std::shared_ptr<Source>>>(mSources);
}

void AudioController::addToUpdateList() {
  mUpdateInstructor->update(shared_from_this());
}

} // namespace cs::audio