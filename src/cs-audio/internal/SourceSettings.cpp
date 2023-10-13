////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "SourceSettings.hpp"

namespace cs::audio {

SourceSettings::SourceSettings() 
  : mUpdateSettings(std::make_shared<std::map<std::string, std::any>>()) 
  , mCurrentSettings(std::make_shared<std::map<std::string, std::any>>()) {
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void SourceSettings::set(std::string key, std::any value) {
  mUpdateSettings->operator[](key) = value;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::shared_ptr<std::map<std::string, std::any>> SourceSettings::getCurrentSettings() const {
  return mCurrentSettings;
}

} // namespace cs::audio