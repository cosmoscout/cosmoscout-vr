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
  std::shared_ptr<ProcessingStepsManager> processingStepsManager) 
  : SourceSettings()
  , mBufferManager(std::move(bufferManager))
  , mProcessingStepsManager(std::move(processingStepsManager))
  , mUpdateBuilder(std::make_shared<UpdateBuilder>()) {
  
  setUpdateBuilder(mUpdateBuilder);  
  mProcessingStepsManager->createPipeline(this);
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

void AudioController::setPipeline(std::vector<std::string> processingSteps) {
  mProcessingStepsManager->createPipeline(processingSteps, this);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void AudioController::update() {
  
  UpdateBuilder::UpdateList updateInstructions = mUpdateBuilder->createUpdateList();

  // updateInstructions.print();

  // update every source and group with plugin settings
  if (updateInstructions.updateAll) {
    updateAll();
    return;
  }

  // update changed groups with member sources
  if (updateInstructions.updateWithGroup.size() > 0) {
    updateGroups(updateInstructions.updateWithGroup);
  }

  // update leftover changed sources
  if (updateInstructions.updateOnlySource.size() > 0) {
    updateSources(updateInstructions.updateOnlySource);
  }
}

void AudioController::updateAll() {
  
  // possible improvement: disable mixing with group settings if there are no group updates -> change in createUpdateList() required

  for (auto sourcePtr : mSources) {
    // TODO: refactor

    // take plugin settings
    std::map<std::string, std::any> x(*mUpdateSettings);
    auto finalSettings = std::make_shared<std::map<std::string, std::any>>(x);

    // add group settings
    if (sourcePtr->mGroup != nullptr) {
      finalSettings = SettingsMixer::OverrideAdd_A_with_B(finalSettings, sourcePtr->mGroup->mUpdateSettings);
    }

    // remove settings that are already set
    finalSettings = SettingsMixer::A_Without_B(finalSettings, sourcePtr->mCurrentSettings);

    // add sourceSettings to finalSettings
    finalSettings = SettingsMixer::OverrideAdd_A_with_B(finalSettings, sourcePtr->mUpdateSettings);

    // run finalSetting through pipeline
    auto failedSettings = mProcessingStepsManager->process(sourcePtr->mOpenAlId, this, finalSettings);

    // Update currently set settings for a source
    sourcePtr->mCurrentSettings = SettingsMixer::OverrideAdd_A_with_B(sourcePtr->mCurrentSettings, SettingsMixer::A_Without_B(finalSettings, failedSettings));
    sourcePtr->mUpdateSettings->clear();
  }

  // Update currently set settings for a group
  for (std::shared_ptr<SourceGroup> groupPtr : mGroups) {
    groupPtr->mCurrentSettings = SettingsMixer::OverrideAdd_A_with_B(groupPtr->mCurrentSettings, groupPtr->mUpdateSettings);
    groupPtr->mUpdateSettings->clear();
  }

  // Update currently set settings for the plugin
  this->mCurrentSettings = SettingsMixer::OverrideAdd_A_with_B(this->mCurrentSettings, this->mUpdateSettings);
  this->mUpdateSettings->clear();
}

void AudioController::updateGroups(std::vector<std::shared_ptr<Source>> sources) {
  
  for (auto sourcePtr : sources) {

    // take group settings
    std::map<std::string, std::any> x(*sourcePtr->mGroup->mUpdateSettings);
    auto finalSettings = std::make_shared<std::map<std::string, std::any>>(x);

    // remove settings that are already set
    finalSettings = SettingsMixer::A_Without_B(finalSettings, sourcePtr->mCurrentSettings);

    // add sourceSettings to finalSettings
    finalSettings = SettingsMixer::OverrideAdd_A_with_B(finalSettings, sourcePtr->mUpdateSettings);

    // run finalSetting through pipeline
    auto failedSettings = mProcessingStepsManager->process(sourcePtr->mOpenAlId, this, finalSettings);
  
    // Update currently set settings for a source
    sourcePtr->mCurrentSettings = SettingsMixer::OverrideAdd_A_with_B(sourcePtr->mCurrentSettings, SettingsMixer::A_Without_B(finalSettings, failedSettings));
    sourcePtr->mUpdateSettings->clear();
  }

  // TODO: update only changed groups
  // Update currently set settings for a group
  for (std::shared_ptr<SourceGroup> group : mGroups) {
    group->mCurrentSettings = SettingsMixer::OverrideAdd_A_with_B(group->mCurrentSettings, group->mUpdateSettings);
    group->mUpdateSettings->clear();
  }
}

void AudioController::updateSources(std::vector<std::shared_ptr<Source>> sources) {

  for (auto sourcePtr : sources) {
    // run finalSetting through pipeline
    auto failedSettings = mProcessingStepsManager->process(sourcePtr->mOpenAlId, this, sourcePtr->mUpdateSettings);
  
    // Update currently set settings for a source
    sourcePtr->mCurrentSettings = SettingsMixer::OverrideAdd_A_with_B(sourcePtr->mCurrentSettings, SettingsMixer::A_Without_B(sourcePtr->mUpdateSettings, failedSettings));
    sourcePtr->mUpdateSettings->clear();
  }
}

void AudioController::addToUpdateList() {
  mUpdateBuilder->updatePlugin();
}

} // namespace cs::audio