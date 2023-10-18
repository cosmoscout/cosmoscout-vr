////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CS_AUDIO_SETTINGS_MIXER_HPP
#define CS_AUDIO_SETTINGS_MIXER_HPP

#include "cs_audio_export.hpp"

#include <memory>
#include <map>
#include <any>
#include <string>

namespace cs::audio {

class CS_AUDIO_EXPORT SettingsMixer {
 public:
  static std::shared_ptr<std::map<std::string, std::any>> A_Without_B(
    std::shared_ptr<std::map<std::string, std::any>> A, 
    std::shared_ptr<std::map<std::string, std::any>> B);

  static std::shared_ptr<std::map<std::string, std::any>> A_Without_B(
    std::shared_ptr<std::map<std::string, std::any>> A, 
    std::shared_ptr<std::vector<std::string>> B);

  static std::shared_ptr<std::map<std::string, std::any>> OverrideAdd_A_with_B(
    std::shared_ptr<std::map<std::string, std::any>> A, 
    std::shared_ptr<std::map<std::string, std::any>> B);
  
 private:

};

} // namespace cs::audio

#endif // CS_AUDIO_SETTINGS_MIXER_HPP
