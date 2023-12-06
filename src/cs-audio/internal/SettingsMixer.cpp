////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "SettingsMixer.hpp"
#include <vector>

namespace cs::audio {

void SettingsMixer::A_Without_B(
  std::shared_ptr<std::map<std::string, std::any>> A, 
  std::shared_ptr<std::map<std::string, std::any>> B) {
  
  for (auto const& [key, val] : *B) {
    if (auto search = A->find(key); search != A->end()) { 
      A->erase(search);
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void SettingsMixer::A_Without_B(
  std::shared_ptr<std::map<std::string, std::any>> A, 
  std::shared_ptr<std::vector<std::string>> B) {
  
  for (auto key : *B) {
    if (auto search = A->find(key); search != A->end()) { 
      A->erase(search);
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void SettingsMixer::A_Without_B_Value(
  std::shared_ptr<std::map<std::string, std::any>> A, 
  std::string B) {
  
  for (auto it = A->begin(); it != A->end(); ++it) {
     if (it->second.type() == typeid(std::string) && std::any_cast<std::string>(it->second) == B) {
      A->erase(it);
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void SettingsMixer::OverrideAdd_A_with_B(
  std::shared_ptr<std::map<std::string, std::any>> A, 
  std::shared_ptr<std::map<std::string, std::any>> B) {
  
  for (auto const& [key, val] : *B) {
    A->operator[](key) = val;
  }
}

void SettingsMixer::Add_A_with_B_if_not_defined(
  std::shared_ptr<std::map<std::string, std::any>> A, 
  std::shared_ptr<std::map<std::string, std::any>> B) {
  
  for (auto const& element : *B) {
    A->insert(element);
  }
}

} // namespace cs::audio