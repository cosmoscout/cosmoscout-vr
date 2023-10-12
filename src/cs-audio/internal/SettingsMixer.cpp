////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "SettingsMixer.hpp"

namespace cs::audio {

////////////////////////////////////////////////////////////////////////////////////////////////////

std::shared_ptr<std::map<std::string, std::any>> SettingsMixer::mixGroupAndSourceUpdate(
  std::shared_ptr<std::map<std::string, std::any>> sourceCurrentSettings,
  std::shared_ptr<std::map<std::string, std::any>> sourceSettings,
  std::shared_ptr<std::map<std::string, std::any>> groupSettings) {
  
  auto result = mixGroupUpdate(sourceCurrentSettings, groupSettings);
  addSettings(*result, sourceSettings);
  return result;
}

std::shared_ptr<std::map<std::string, std::any>> SettingsMixer::mixGroupUpdate(
  std::shared_ptr<std::map<std::string, std::any>> sourceCurrentSettings,
  std::shared_ptr<std::map<std::string, std::any>> groupSettings) {
  
  // only set groupSettings that are not already set in sourceCurrentSettings
  auto result = std::make_shared<std::map<std::string, std::any>>();
  for (auto const& [key, val] : *groupSettings) {
    if (auto search = sourceCurrentSettings->find(key); search != sourceCurrentSettings->end()) { 
      continue;
    }
    result->operator[](key) = val;
  }
  return result;
}

void SettingsMixer::addSettings(std::map<std::string, std::any> &baseSettings,
  std::shared_ptr<std::map<std::string, std::any>> newSettings) {
  
  for (auto const& [key, val] : *newSettings) {
    baseSettings[key] = val;
  }
}

} // namespace cs::audio