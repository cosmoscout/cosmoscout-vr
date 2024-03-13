////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "AudioController.hpp"
#include "../cs-utils/FrameStats.hpp"
#include "Source.hpp"
#include "SourceGroup.hpp"
#include "StreamingSource.hpp"
#include "internal/BufferManager.hpp"
#include "internal/ProcessingStepsManager.hpp"
#include "internal/SettingsMixer.hpp"
#include "internal/UpdateInstructor.hpp"

namespace cs::audio {

AudioController::AudioController(std::shared_ptr<BufferManager> bufferManager,
    std::shared_ptr<ProcessingStepsManager>                     processingStepsManager,
    std::shared_ptr<UpdateConstructor> updateConstructor, int id)
    : SourceSettings()
    , std::enable_shared_from_this<AudioController>()
    , mControllerId(id)
    , mBufferManager(std::move(bufferManager))
    , mProcessingStepsManager(std::move(processingStepsManager))
    , mSources(std::vector<std::weak_ptr<SourceBase>>())
    , mStreams(std::vector<std::weak_ptr<StreamingSource>>())
    , mGroups(std::vector<std::weak_ptr<SourceGroup>>())
    , mUpdateInstructor(std::make_shared<UpdateInstructor>())
    , mUpdateConstructor(std::move(updateConstructor)) {
  setUpdateInstructor(mUpdateInstructor);
}

AudioController::AudioController()
    : SourceSettings(false)
    , std::enable_shared_from_this<AudioController>()
    , mControllerId(-1) {
}

AudioController::~AudioController() {
  if (mIsLeader) {
    mProcessingStepsManager->removeAudioController(mControllerId);
    mSources.clear();
    mStreams.clear();
    mGroups.clear();
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::shared_ptr<SourceGroup> AudioController::createSourceGroup() {
  if (!mIsLeader) {
    return std::make_shared<SourceGroup>();
  }
  auto group =
      std::make_shared<SourceGroup>(mUpdateInstructor, mUpdateConstructor, shared_from_this());
  mGroups.push_back(group);
  return group;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::shared_ptr<Source> AudioController::createSource(std::string file) {
  if (!mIsLeader) {
    return std::make_shared<Source>();
  }
  auto source = std::make_shared<Source>(mBufferManager, file, mUpdateInstructor);
  mSources.push_back(source);

  // apply audioController settings to newly creates source
  if (!mCurrentSettings->empty()) {
    mUpdateConstructor->applyCurrentControllerSettings(
        source, shared_from_this(), mCurrentSettings);
  }
  return source;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::shared_ptr<StreamingSource> AudioController::createStreamingSource(
    std::string file, int bufferSize, int queueSize) {

  if (!mIsLeader) {
    return std::make_shared<StreamingSource>();
  }
  auto source = std::make_shared<StreamingSource>(file, bufferSize, queueSize, mUpdateInstructor);
  mSources.push_back(source);
  mStreams.push_back(source);

  // apply audioController settings to newly creates source
  if (!mCurrentSettings->empty()) {
    mUpdateConstructor->applyCurrentControllerSettings(
        source, shared_from_this(), mCurrentSettings);
  }
  return source;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void AudioController::setPipeline(std::vector<std::string> processingSteps) {
  if (!mIsLeader) {
    return;
  }
  mProcessingStepsManager->createPipeline(processingSteps, mControllerId);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void AudioController::update() {
  if (!mIsLeader) {
    return;
  }
  auto frameStats = cs::utils::FrameStats::ScopedTimer(
      "AudioEngineController", cs::utils::FrameStats::TimerMode::eCPU);

  auto updateInstructions = mUpdateInstructor->createUpdateInstruction();

  // update every source and group with plugin settings
  if (updateInstructions.updateAll) {
    mUpdateConstructor->updateAll(
        std::make_shared<std::vector<std::shared_ptr<SourceBase>>>(getSources()),
        std::make_shared<std::vector<std::shared_ptr<SourceGroup>>>(getGroups()),
        shared_from_this());
    return;
  }

  // update changed groups with member sources
  if (updateInstructions.updateWithGroup->size() > 0) {
    mUpdateConstructor->updateGroups(updateInstructions.updateWithGroup,
        std::make_shared<std::vector<std::shared_ptr<SourceGroup>>>(getGroups()),
        shared_from_this());
  }

  // update leftover changed sources
  if (updateInstructions.updateSourceOnly->size() > 0) {
    mUpdateConstructor->updateSources(updateInstructions.updateSourceOnly, shared_from_this());
  }
}

void AudioController::updateStreamingSources() {
  if (!mIsLeader) {
    return;
  }
  bool streamExpired = false;
  for (auto stream : mStreams) {
    if (stream.expired()) {
      streamExpired = true;
      continue;
    }
    if (stream.lock()->updateStream()) {
      update();
    }
  }
  if (streamExpired) {
    removeExpiredElements<StreamingSource>(mStreams);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::vector<std::shared_ptr<SourceBase>> AudioController::getSources() {
  if (!mIsLeader) {
    std::vector<std::shared_ptr<SourceBase>>();
  }
  std::vector<std::shared_ptr<SourceBase>> sourcesShared;
  bool                                     sourceExpired = false;

  for (auto source : mSources) {
    if (source.expired()) {
      sourceExpired = true;
      continue;
    }
    sourcesShared.push_back(source.lock());
  }

  if (sourceExpired) {
    removeExpiredElements<SourceBase>(mSources);
  }

  return sourcesShared;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::vector<std::shared_ptr<SourceGroup>> AudioController::getGroups() {
  if (!mIsLeader) {
    std::vector<std::shared_ptr<SourceGroup>>();
  }
  std::vector<std::shared_ptr<SourceGroup>> groupsShared;
  bool                                      groupExpired = false;

  for (auto group : mGroups) {
    if (group.expired()) {
      groupExpired = true;
      continue;
    }
    groupsShared.push_back(group.lock());
  }

  if (groupExpired) {
    removeExpiredElements<SourceGroup>(mGroups);
  }
  return groupsShared;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
void AudioController::removeExpiredElements(std::vector<std::weak_ptr<T>> elements) {
  elements.erase(std::remove_if(elements.begin(), elements.end(),
                     [](const std::weak_ptr<T>& ptr) { return ptr.expired(); }),
      elements.end());
}

////////////////////////////////////////////////////////////////////////////////////////////////////

const int AudioController::getControllerId() const {
  if (!mIsLeader) {
    return 0;
  }
  return mControllerId;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void AudioController::addToUpdateList() {
  mUpdateInstructor->update(shared_from_this());
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void AudioController::removeFromUpdateList() {
  mUpdateInstructor->removeUpdate(shared_from_this());
}

} // namespace cs::audio