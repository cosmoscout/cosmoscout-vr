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

#include <vector>
#include <string>

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
  std::shared_ptr<AudioController> audioController) {

  // possible improvement: disable mixing with group settings if there are no group updates -> change in createUpdateList() required

  if (containsRemove(audioController->mUpdateSettings)) {
    for (auto sourcePtr : *sources) {
      rebuildPlaybackSettings(audioController, sourcePtr);
    }
    goto sourcesProcessed;
  }

  for (auto sourcePtr : *sources) {

    if (containsRemove(sourcePtr->mUpdateSettings)) {
      rebuildPlaybackSettings(audioController, sourcePtr);
      continue;
    }

    if (sourcePtr->mGroup != nullptr && containsRemove(sourcePtr->mGroup->mUpdateSettings)) {
      rebuildPlaybackSettings(audioController, sourcePtr);
      continue;
    }

    // take controller settings
    auto finalSettings = std::make_shared<std::map<std::string, std::any>>(*(audioController->mUpdateSettings));

    // remove controller settings that are already set by the source
    SettingsMixer::A_Without_B(finalSettings, sourcePtr->mCurrentSettings);
 
    if (sourcePtr->mGroup != nullptr) {
      // remove controller settings that are already set by the group
      SettingsMixer::A_Without_B(finalSettings, sourcePtr->mGroup->mCurrentSettings);

      // take group settings
      auto finalGroup = std::make_shared<std::map<std::string, std::any>>(*(sourcePtr->mGroup->mUpdateSettings));

      // remove group settings that are already set by the source
      SettingsMixer::A_Without_B(finalGroup, sourcePtr->mCurrentSettings);

      // Mix controller and group Settings
      SettingsMixer::OverrideAdd_A_with_B(finalSettings, finalGroup);
    }

    // add source update settings to finalSettings
    SettingsMixer::OverrideAdd_A_with_B(finalSettings, sourcePtr->mUpdateSettings);

    // run finalSetting through pipeline
    auto failedSettings = mProcessingStepsManager->process(sourcePtr, audioController, finalSettings);

    // update current source playback settings 
    SettingsMixer::A_Without_B(finalSettings, failedSettings);
    SettingsMixer::OverrideAdd_A_with_B(sourcePtr->mPlaybackSettings, finalSettings);

    // Update current source settings
    SettingsMixer::A_Without_B(sourcePtr->mUpdateSettings, failedSettings);
    SettingsMixer::OverrideAdd_A_with_B(sourcePtr->mCurrentSettings, sourcePtr->mUpdateSettings);
    sourcePtr->mUpdateSettings->clear();
  }

  sourcesProcessed:
  // Update currently set settings for a group
  for (std::shared_ptr<SourceGroup> groupPtr : *groups) {
    if (!groupPtr->mUpdateSettings->empty()) {
      SettingsMixer::OverrideAdd_A_with_B(groupPtr->mCurrentSettings, groupPtr->mUpdateSettings);
      groupPtr->mUpdateSettings->clear();
    }
  }

  // Update currently set settings for the plugin
  SettingsMixer::OverrideAdd_A_with_B(audioController->mCurrentSettings, audioController->mUpdateSettings);
  audioController->mUpdateSettings->clear();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void UpdateConstructor::updateGroups(
  std::shared_ptr<std::vector<std::shared_ptr<Source>>> sources,
  std::shared_ptr<std::vector<std::shared_ptr<SourceGroup>>> groups,
  std::shared_ptr<AudioController> audioController) {

  for (auto sourcePtr : *sources) {

    if (containsRemove(sourcePtr->mUpdateSettings) || containsRemove(sourcePtr->mGroup->mUpdateSettings)) {
      rebuildPlaybackSettings(audioController, sourcePtr);
      continue;
    }

    // take group settings
    auto finalSettings = std::make_shared<std::map<std::string, std::any>>(*(sourcePtr->mGroup->mUpdateSettings));

    // remove settings that are already set by the source
    SettingsMixer::A_Without_B(finalSettings, sourcePtr->mCurrentSettings);

    // add source update settings to finalSettings
    SettingsMixer::OverrideAdd_A_with_B(finalSettings, sourcePtr->mUpdateSettings);

    // run finalSetting through pipeline
    auto failedSettings = mProcessingStepsManager->process(sourcePtr, audioController, finalSettings);

    // update current source playback settings 
    SettingsMixer::A_Without_B(finalSettings, failedSettings);
    SettingsMixer::OverrideAdd_A_with_B(sourcePtr->mPlaybackSettings, finalSettings);

    // Update current source settings 
    SettingsMixer::A_Without_B(sourcePtr->mUpdateSettings, failedSettings);
    SettingsMixer::OverrideAdd_A_with_B(sourcePtr->mCurrentSettings, sourcePtr->mUpdateSettings);
    sourcePtr->mUpdateSettings->clear();
  }

  // Update currently set settings for a group
  for (std::shared_ptr<SourceGroup> groupPtr : *groups) {
    if (!groupPtr->mUpdateSettings->empty()) {
      SettingsMixer::OverrideAdd_A_with_B(groupPtr->mCurrentSettings, groupPtr->mUpdateSettings);
      groupPtr->mUpdateSettings->clear();
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void UpdateConstructor::updateSources(
  std::shared_ptr<std::vector<std::shared_ptr<Source>>> sources,
  std::shared_ptr<AudioController> audioController) {

  for (auto sourcePtr : *sources) {

    if (containsRemove(sourcePtr->mUpdateSettings)) {
      rebuildPlaybackSettings(audioController, sourcePtr);
      continue;
    }

    // run finalSetting through pipeline
    auto failedSettings = mProcessingStepsManager->process(sourcePtr, audioController, sourcePtr->mUpdateSettings);

    // update current source playback settings 
    SettingsMixer::A_Without_B(sourcePtr->mUpdateSettings, failedSettings);
    SettingsMixer::OverrideAdd_A_with_B(sourcePtr->mPlaybackSettings, sourcePtr->mUpdateSettings);

    // Update currently set settings for a source
    SettingsMixer::OverrideAdd_A_with_B(sourcePtr->mCurrentSettings, sourcePtr->mUpdateSettings);
    sourcePtr->mUpdateSettings->clear();
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void UpdateConstructor::applyCurrentControllerSettings(
  std::shared_ptr<Source> source,
  std::shared_ptr<AudioController> audioController,
  std::shared_ptr<std::map<std::string, std::any>> settings) {
  
  // There is no need to check for already set values here because this functions only gets called when creating a new 
  // source, at which point there cannot be any previous settings.

  // run finalSetting through pipeline
  auto failedSettings = mProcessingStepsManager->process(source, audioController, settings);
  
  // Update currently set settings for a source
  auto settingsCopy(settings);
  SettingsMixer::A_Without_B(settingsCopy, failedSettings);
  SettingsMixer::OverrideAdd_A_with_B(source->mPlaybackSettings, settingsCopy);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void UpdateConstructor::applyCurrentGroupSettings(
  std::shared_ptr<Source> source,
  std::shared_ptr<AudioController> audioController,
  std::shared_ptr<std::map<std::string, std::any>> settings) {
  
  // take group settings
  std::map<std::string, std::any> x(*settings);
  auto finalSettings = std::make_shared<std::map<std::string, std::any>>(x);

  // remove settings that are already set
  SettingsMixer::A_Without_B(finalSettings, source->mCurrentSettings);

  // run finalSetting through pipeline
  auto failedSettings = mProcessingStepsManager->process(source, audioController, finalSettings);

  // Update currently set settings for a source
  SettingsMixer::A_Without_B(finalSettings, failedSettings);
  SettingsMixer::OverrideAdd_A_with_B(source->mPlaybackSettings, finalSettings);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool UpdateConstructor::containsRemove(std::shared_ptr<std::map<std::string, std::any>> settings) {
  for (auto it = settings->begin(); it != settings->end(); ++it) {
     if (it->second.type() == typeid(std::string) && std::any_cast<std::string>(it->second) == "remove") {
      return true;
    }
  }
  return false;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void UpdateConstructor::rebuildPlaybackSettings(std::shared_ptr<AudioController> audioController, std::shared_ptr<Source> source) {
  // take current controller settings
  auto finalSettings = std::make_shared<std::map<std::string, std::any>>(*(audioController->mCurrentSettings));
  // Mix with controller update settings
  SettingsMixer::OverrideAdd_A_with_B(finalSettings, audioController->mUpdateSettings);

  // update current controller settings
  auto controllerSettings = std::make_shared<std::map<std::string, std::any>>(*(finalSettings));

  // removing "remove" because "remove" settings should not appear in the current settings 
  SettingsMixer::A_Without_B_Value(controllerSettings, "remove");
  audioController->mCurrentSettings = controllerSettings;
  audioController->mUpdateSettings->clear();

  if (source->mGroup != nullptr) {
    // take current group settings
    auto groupSettings = std::make_shared<std::map<std::string, std::any>>(*(source->mGroup->mCurrentSettings));

    // Mix with group update Settings   
    SettingsMixer::OverrideAdd_A_with_B(groupSettings, source->mGroup->mUpdateSettings);

    // filter out remove settings
    auto normalGroupSettings = std::make_shared<std::map<std::string, std::any>>(*(groupSettings));
    SettingsMixer::A_Without_B_Value(normalGroupSettings, "remove");

    // create remove settings 
    auto removeGroupSettings = std::make_shared<std::map<std::string, std::any>>(*(groupSettings));
    SettingsMixer::A_Without_B(removeGroupSettings, normalGroupSettings);

    // add controller settings with normalSettings
    SettingsMixer::OverrideAdd_A_with_B(finalSettings, normalGroupSettings); 

    // add finalSettings with removeSettings
    SettingsMixer::Add_A_with_B_if_not_defined(finalSettings, removeGroupSettings);

    // update current group settings
    source->mGroup->mCurrentSettings = normalGroupSettings;
    source->mGroup->mUpdateSettings->clear();
  }

  // take current source settings
  auto sourceSettings = std::make_shared<std::map<std::string, std::any>>(*(source->mCurrentSettings));

  // Mix with source update Settings   
  SettingsMixer::OverrideAdd_A_with_B(sourceSettings, source->mUpdateSettings);

  // filter out remove settings
  auto normalSourceSettings = std::make_shared<std::map<std::string, std::any>>(*(sourceSettings));
  SettingsMixer::A_Without_B_Value(normalSourceSettings, "remove");

  // create remove settings 
  auto removeSourceSettings = std::make_shared<std::map<std::string, std::any>>(*(sourceSettings));
  SettingsMixer::A_Without_B(removeSourceSettings, normalSourceSettings);

  // add finalSettings with normalSettings
  SettingsMixer::OverrideAdd_A_with_B(finalSettings, normalSourceSettings);

  // add finalSettings with removeSettings
  SettingsMixer::Add_A_with_B_if_not_defined(finalSettings, removeSourceSettings); 

  // run finalSettings through pipeline
  auto failedSettings = mProcessingStepsManager->process(source, audioController, finalSettings);

  // Update current playback settings for a source
  SettingsMixer::A_Without_B(finalSettings, failedSettings);
  SettingsMixer::A_Without_B_Value(finalSettings, "remove");
  SettingsMixer::OverrideAdd_A_with_B(source->mPlaybackSettings, finalSettings);

  // Update currently set settings by a source
  SettingsMixer::A_Without_B(normalSourceSettings, failedSettings);
  source->mCurrentSettings = normalSourceSettings;
  source->mUpdateSettings->clear();
}

} // namespace cs::audio