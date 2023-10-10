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
  /// Mixes groupSettings with sourceCurrentSettings so that only groupSettings
  /// are set that are not already set in sourceCurrentSettings and overrides
  /// groupSettings with sourcSettings if they are set. This mix function 
  /// is used in SourceGroup.updateAll().
  static std::shared_ptr<std::map<std::string, std::any>> mixGroupAndSourceUpdate(
    std::shared_ptr<std::map<std::string, std::any>> sourceCurrentSettings,
    std::shared_ptr<std::map<std::string, std::any>> sourceSettings,
    std::shared_ptr<std::map<std::string, std::any>> groupSettings);

  /// Mixes sourceCurrentSettings with groupSettings so that only groupSettings
  /// are set that are not already set in sourceCurrentSettings. This mix function 
  /// is used in SourceGroup.update().
  static std::shared_ptr<std::map<std::string, std::any>> mixGroupUpdate(
    std::shared_ptr<std::map<std::string, std::any>> sourceCurrentSettings,
    std::shared_ptr<std::map<std::string, std::any>> groupSettings);

  // Add/override newSettings to the existing currentSettings
  static void addSettings(
    std::map<std::string, std::any> &currentSettings,
    std::shared_ptr<std::map<std::string, std::any>> newSettings);

 private:

};

} // namespace cs::audio

#endif // CS_AUDIO_SETTINGS_MIXER_HPP
