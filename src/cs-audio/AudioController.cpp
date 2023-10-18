////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "AudioController.hpp"
#include "internal/BufferManager.hpp"
#include "internal/ProcessingStepsManager.hpp"
#include "internal/SettingsMixer.hpp"
#include "internal/UpdateBuilder.hpp"
#include "Source.hpp"
#include "SourceGroup.hpp"

namespace cs::audio {

AudioController::AudioController(
  std::shared_ptr<BufferManager> bufferManager, 
  std::shared_ptr<ProcessingStepsManager> processingStepsManager,
  std::vector<std::string> processingSteps) 
  : SourceSettings()
  , mBufferManager(std::move(bufferManager))
  , mProcessingStepsManager(std::move(processingStepsManager))
  , mUpdateBuilder(std::make_shared<UpdateBuilder>()) {
  
  setUpdateBuilder(mUpdateBuilder);

  // TODO: define pipeline via config file
  mProcessingStepsManager->createPipeline(processingSteps, mAudioControllerId);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::shared_ptr<SourceGroup> AudioController::createSourceGroup() {
  auto group = std::make_shared<SourceGroup>(mUpdateBuilder);
  mGroups.push_back(group);
  return group;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::shared_ptr<Source> AudioController::createSource(std::string file) {
  auto source = std::make_shared<Source>(mBufferManager, mProcessingStepsManager, file, mUpdateBuilder);
  mSources.push_back(source);
  return source;
} 

////////////////////////////////////////////////////////////////////////////////////////////////////

void AudioController::update() {
  
  for (std::shared_ptr<Source> source : mSources) {
    // TODO: refactor
    
    // take plugin settings
    std::map<std::string, std::any> x(*mUpdateSettings);
    auto finalSettings = std::make_shared<std::map<std::string, std::any>>(x);

    // add group settings
    if (source->mGroup != nullptr) {
      finalSettings = SettingsMixer::OverrideAdd_A_with_B(finalSettings, source->mGroup->mUpdateSettings);
    }

    // skip if there is nothing to update
    if (finalSettings->empty() && source->mUpdateSettings->empty()) {
      continue;
    }
    
    // remove settings that are already set
    finalSettings = SettingsMixer::A_Without_B(finalSettings, source->mCurrentSettings);

    // add sourceSettings to finalSettings
    finalSettings = SettingsMixer::OverrideAdd_A_with_B(finalSettings, source->mUpdateSettings);

    // run finalSetting through pipeline
    mProcessingStepsManager->process(source->mOpenAlId, std::shared_ptr<AudioController>(this), finalSettings);
  
    // Update currently set settings for a source
    source->mCurrentSettings = SettingsMixer::OverrideAdd_A_with_B(source->mCurrentSettings, finalSettings);
    source->mUpdateSettings->clear();
  }

  // Update currently set settings for a group
  for (std::shared_ptr<SourceGroup> group : mGroups) {
    group->mCurrentSettings = SettingsMixer::OverrideAdd_A_with_B(group->mCurrentSettings, group->mUpdateSettings);
    group->mUpdateSettings->clear();
  }

  // Update currently set settings for the plugin
  this->mCurrentSettings = SettingsMixer::OverrideAdd_A_with_B(this->mCurrentSettings, this->mUpdateSettings);
  this->mUpdateSettings->clear();
}

} // namespace cs::audio