////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "SourceGroup.hpp"
#include "Source.hpp"
#include "internal/SettingsMixer.hpp"

namespace cs::audio {

SourceGroup::SourceGroup(std::shared_ptr<ProcessingStepsManager> processingStepsManager) 
  : mProcessingStepsManager(std::move(processingStepsManager))
  , mSettings(std::make_shared<std::map<std::string, std::any>>())
  , mCurrentSettings(std::make_shared<std::map<std::string, std::any>>()) 
  , mMemberSources(std::set<std::shared_ptr<Source>>()) {
}

void SourceGroup::add(std::shared_ptr<Source> source) {
  mMemberSources.insert(source);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void SourceGroup::remove(std::shared_ptr<Source> sourceToRemove) {
  mMemberSources.erase(sourceToRemove);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void SourceGroup::reset() {
  mMemberSources.clear();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void SourceGroup::update() {
  if (this->mSettings->empty()) {
    return;
  }
  for (std::shared_ptr<Source> source : mMemberSources) {
    // mix group and source settings
    auto settingsToSet = SettingsMixer::mixGroupUpdate(source->mCurrentSettings, this->mSettings);
    
    // process new settings and update current source settings
    mProcessingStepsManager->process(source->mOpenAlId, settingsToSet);
    // TODO: ErrorHandling
    SettingsMixer::addSettings(*(source->mCurrentSettings), settingsToSet);
    
    // reset source settings
    source->mSettings->clear();
  }
  // update current group settings
  SettingsMixer::addSettings(*(this->mCurrentSettings), this->mSettings);

  // reset group settings
  this->mSettings->clear();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void SourceGroup::updateAll() {
  if (this->mSettings->empty()) {
    updateMembersOnly();
    return;
  }
  for (std::shared_ptr<Source> source : mMemberSources) {
    // mix group and source settings
    auto settingsToSet = SettingsMixer::mixGroupAndSourceUpdate(source->mCurrentSettings, 
      source->mSettings, this->mSettings);

    // process new settings and update current source settings
    mProcessingStepsManager->process(source->mOpenAlId, settingsToSet);    
    // TODO: ErrorHandling
    SettingsMixer::addSettings(*(source->mCurrentSettings), settingsToSet);

    // reset source settings
    source->mSettings->clear();
  }
  // update current group settings
  SettingsMixer::addSettings(*(this->mCurrentSettings), this->mSettings);
  
  // reset group settings
  this->mSettings->clear();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void SourceGroup::updateMembersOnly() {
  for (std::shared_ptr<Source> source : mMemberSources) {
    source->update();
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void SourceGroup::set(std::string key, std::any value) {
  mSettings->operator[](key) = value;
}

} // namespace cs::audio