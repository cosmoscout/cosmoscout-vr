////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "SourceSettings.hpp"

namespace cs::audio {

SourceSettings::SourceSettings(std::shared_ptr<UpdateBuilder> updateBuilder) 
  : mUpdateSettings(std::make_shared<std::map<std::string, std::any>>()) 
  , mCurrentSettings(std::make_shared<std::map<std::string, std::any>>())
  , mUpdateBuilder(std::move(updateBuilder)) {
}

////////////////////////////////////////////////////////////////////////////////////////////////////

SourceSettings::SourceSettings() 
  : mUpdateSettings(std::make_shared<std::map<std::string, std::any>>()) 
  , mCurrentSettings(std::make_shared<std::map<std::string, std::any>>()) {
}

void SourceSettings::setUpdateBuilder(std::shared_ptr<UpdateBuilder> updateBuilder) {
  mUpdateBuilder = updateBuilder;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void SourceSettings::set(std::string key, std::any value) {
  mUpdateSettings->operator[](key) = value;
  addToUpdateList();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::shared_ptr<std::map<std::string, std::any>> SourceSettings::getCurrentSettings() const {
  return mCurrentSettings;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void SourceSettings::remove(std::string key) {
  mCurrentSettings->erase(key);
  mUpdateSettings->erase(key);
  addToUpdateList();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void SourceSettings::removeUpdate(std::string key) {
  mUpdateSettings->erase(key);
}

} // namespace cs::audio