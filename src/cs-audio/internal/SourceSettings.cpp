////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "SourceSettings.hpp"
#include <string>
#include <iostream>

namespace cs::audio {

SourceSettings::SourceSettings(std::shared_ptr<UpdateInstructor> UpdateInstructor) 
  : mIsLeader(true)
  , mUpdateSettings(std::make_shared<std::map<std::string, std::any>>()) 
  , mCurrentSettings(std::make_shared<std::map<std::string, std::any>>())
  , mUpdateInstructor(std::move(UpdateInstructor)) {
}

////////////////////////////////////////////////////////////////////////////////////////////////////

SourceSettings::SourceSettings()
  : mIsLeader(true)
  , mUpdateSettings(std::make_shared<std::map<std::string, std::any>>()) 
  , mCurrentSettings(std::make_shared<std::map<std::string, std::any>>()) {
}

////////////////////////////////////////////////////////////////////////////////////////////////////

SourceSettings::SourceSettings(bool isLeader) 
  : mIsLeader(false) {
}

////////////////////////////////////////////////////////////////////////////////////////////////////

SourceSettings::~SourceSettings() {
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void SourceSettings::setUpdateInstructor(std::shared_ptr<UpdateInstructor> UpdateInstructor) {
  if (!mIsLeader) { return; }
  mUpdateInstructor = UpdateInstructor;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void SourceSettings::set(std::string key, std::any value) {
  if (!mIsLeader) { return; }
  mUpdateSettings->operator[](key) = value;
  addToUpdateList();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

const std::shared_ptr<const std::map<std::string, std::any>> SourceSettings::getCurrentSettings() const {
  if (!mIsLeader) { std::shared_ptr<const std::map<std::string, std::any>>(); }
  return mCurrentSettings;
}

const std::shared_ptr<const std::map<std::string, std::any>> SourceSettings::getUpdateSettings() const {
  if (!mIsLeader) { std::shared_ptr<const std::map<std::string, std::any>>(); }
  return mUpdateSettings;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void SourceSettings::remove(std::string key) {
  if (!mIsLeader) { return; }
  mUpdateSettings->erase(key);
  if (mCurrentSettings->find(key) == mCurrentSettings->end()) {
    return;
  }
  mUpdateSettings->operator[](key) = std::string("remove");
  addToUpdateList();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void SourceSettings::removeUpdate(std::string key) {
  if (!mIsLeader) { return; }
  mUpdateSettings->erase(key);
  if (mUpdateSettings->empty()) {
    removeFromUpdateList();
  }
}

} // namespace cs::audio