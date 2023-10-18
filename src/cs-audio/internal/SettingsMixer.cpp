////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "SettingsMixer.hpp"
#include <vector>

namespace cs::audio {

std::shared_ptr<std::map<std::string, std::any>> SettingsMixer::A_Without_B(
  std::shared_ptr<std::map<std::string, std::any>> A, 
  std::shared_ptr<std::map<std::string, std::any>> B) {
  
  auto result = std::make_shared<std::map<std::string, std::any>>();
  for (auto const& [key, val] : *A) {
    if (auto search = B->find(key); search == B->end()) { 
      result->operator[](key) = val;
    }
  }
  return result;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::shared_ptr<std::map<std::string, std::any>> SettingsMixer::A_Without_B(
  std::shared_ptr<std::map<std::string, std::any>> A, 
  std::shared_ptr<std::vector<std::string>> B) {
  
  auto result = std::make_shared<std::map<std::string, std::any>>();
  for (auto const& [key, val] : *A) {
    
    if (std::find(B->begin(), B->end(), key) == B->end()) {
      result->operator[](key) = val;
    }
  }
  return result;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::shared_ptr<std::map<std::string, std::any>> SettingsMixer::OverrideAdd_A_with_B(
  std::shared_ptr<std::map<std::string, std::any>> A, 
  std::shared_ptr<std::map<std::string, std::any>> B) {
  
  std::map<std::string, std::any> result(*A);
  for (auto const& [key, val] : *B) {
    result[key] = val;
  }
  return std::make_shared<std::map<std::string, std::any>>(result);
}

} // namespace cs::audio