////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "UpdateConstructor.hpp"
#include "SettingsMixer.hpp"
#include "../AudioController.hpp"
#include "../Source.hpp"
#include "../SourceGroup.hpp"

namespace cs::audio {

std::shared_ptr<UpdateConstructor> UpdateConstructor::createUpdateConstructor(
  std::shared_ptr<ProcessingStepsManager> processingStepsManager) {
    
  static auto updateConstructor = std::shared_ptr<UpdateConstructor>(
    new UpdateConstructor(processingStepsManager));
  return updateConstructor;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

UpdateConstructor::UpdateConstructor(std::shared_ptr<ProcessingStepsManager> processingStepsManager) 
  : mProcessingStepsManager(std::move(processingStepsManager)) {
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void UpdateConstructor::updateAll(
  std::shared_ptr<std::vector<std::shared_ptr<Source>>> sources, 
  std::shared_ptr<std::vector<std::shared_ptr<SourceGroup>>> groups,
  AudioController* audioController) {
  
  // possible improvement: disable mixing with group settings if there are no group updates -> change in createUpdateList() required

  for (auto sourcePtr : *sources) {
    // TODO: refactor

    // take plugin settings
    std::map<std::string, std::any> x(*audioController->mUpdateSettings);
    auto finalSettings = std::make_shared<std::map<std::string, std::any>>(x);  

    // add group settings
    if (sourcePtr->mGroup != nullptr) {
      SettingsMixer::OverrideAdd_A_with_B(finalSettings, sourcePtr->mGroup->mUpdateSettings);
    }

    // remove settings that are already set
    SettingsMixer::A_Without_B(finalSettings, sourcePtr->mCurrentSettings);

    // add sourceSettings to finalSettings
    SettingsMixer::OverrideAdd_A_with_B(finalSettings, sourcePtr->mUpdateSettings);

    // run finalSetting through pipeline
    auto failedSettings = mProcessingStepsManager->process(sourcePtr->getOpenAlId(), audioController, finalSettings);

    // Update currently set settings for a source
    SettingsMixer::A_Without_B(finalSettings, failedSettings);
    SettingsMixer::OverrideAdd_A_with_B(sourcePtr->mCurrentSettings, finalSettings);
    sourcePtr->mUpdateSettings->clear();
  }

  // Update currently set settings for a group
  for (std::shared_ptr<SourceGroup> groupPtr : *groups) {
    SettingsMixer::OverrideAdd_A_with_B(groupPtr->mCurrentSettings, groupPtr->mUpdateSettings);
    groupPtr->mUpdateSettings->clear();
  }

  // Update currently set settings for the plugin
  SettingsMixer::OverrideAdd_A_with_B(audioController->mCurrentSettings, audioController->mUpdateSettings);
  audioController->mUpdateSettings->clear();
}

void UpdateConstructor::updateGroups(
  std::shared_ptr<std::vector<std::shared_ptr<Source>>> sources,
  std::shared_ptr<std::vector<std::shared_ptr<SourceGroup>>> groups,
  AudioController* audioController) {
  
  for (auto sourcePtr : *sources) {

    // take group settings
    std::map<std::string, std::any> x(*sourcePtr->mGroup->mUpdateSettings);
    auto finalSettings = std::make_shared<std::map<std::string, std::any>>(x);

    // remove settings that are already set
    SettingsMixer::A_Without_B(finalSettings, sourcePtr->mCurrentSettings);

    // add sourceSettings to finalSettings
    SettingsMixer::OverrideAdd_A_with_B(finalSettings, sourcePtr->mUpdateSettings);

    // run finalSetting through pipeline
    auto failedSettings = mProcessingStepsManager->process(sourcePtr->getOpenAlId(), audioController, finalSettings);
  
    // Update currently set settings for a source
    SettingsMixer::A_Without_B(finalSettings, failedSettings);
    SettingsMixer::OverrideAdd_A_with_B(sourcePtr->mCurrentSettings, finalSettings);
    sourcePtr->mUpdateSettings->clear();
  }

  // Update currently set settings for a group
  for (std::shared_ptr<SourceGroup> group : *groups) {
    if (!group->mUpdateSettings->empty()) {
      SettingsMixer::OverrideAdd_A_with_B(group->mCurrentSettings, group->mUpdateSettings);
      group->mUpdateSettings->clear();
    }
  }
}

void UpdateConstructor::updateSources(
  std::shared_ptr<std::vector<std::shared_ptr<Source>>> sources,
  AudioController* audioController) {

  for (auto sourcePtr : *sources) {
    // run finalSetting through pipeline
    auto failedSettings = mProcessingStepsManager->process(sourcePtr->getOpenAlId(), audioController, sourcePtr->mUpdateSettings);
  
    // Update currently set settings for a source
    SettingsMixer::A_Without_B(sourcePtr->mUpdateSettings, failedSettings);
    SettingsMixer::OverrideAdd_A_with_B(sourcePtr->mCurrentSettings, sourcePtr->mUpdateSettings);
    sourcePtr->mUpdateSettings->clear();
  }
}

void UpdateConstructor::applyCurrentControllerSettings(
  std::shared_ptr<Source> source,
  AudioController* audioController,
  std::shared_ptr<std::map<std::string, std::any>> settings) {
  
  // There is no need to check for already set values here because this functions only gets called when creating a new 
  // source, at which point there cannot be any already set settings.

  // run finalSetting through pipeline
  auto failedSettings = mProcessingStepsManager->process(source->getOpenAlId(), audioController, settings);
  
  // Update currently set settings for a source
  auto x(settings);
  SettingsMixer::A_Without_B(x, failedSettings);
  SettingsMixer::OverrideAdd_A_with_B(source->mCurrentSettings, x);
}

void UpdateConstructor::applyCurrentGroupSettings(
  std::shared_ptr<Source> source,
  AudioController* audioController,
  std::shared_ptr<std::map<std::string, std::any>> settings) {
  
  // take group settings
  std::map<std::string, std::any> x(*settings);
  auto finalSettings = std::make_shared<std::map<std::string, std::any>>(x);

  // remove settings that are already set
  SettingsMixer::A_Without_B(finalSettings, source->mCurrentSettings);

  // run finalSetting through pipeline
  auto failedSettings = mProcessingStepsManager->process(source->getOpenAlId(), audioController, finalSettings);

  // Update currently set settings for a source
  SettingsMixer::A_Without_B(finalSettings, failedSettings);
  SettingsMixer::OverrideAdd_A_with_B(source->mCurrentSettings, finalSettings);
}

} // namespace cs::audio