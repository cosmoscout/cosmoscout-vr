////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CS_AUDIO_SOURCE_SETTINGS_HPP
#define CS_AUDIO_SOURCE_SETTINGS_HPP

#include "cs_audio_export.hpp"

#include <map>
#include <any>
#include <string>
#include <memory>

namespace cs::audio {

class CS_AUDIO_EXPORT SourceSettings {
 public:
  /// Sets a value in mUpdateSettings 
  void set(std::string key, std::any value);
  /// Returns the currently set settings 
  std::shared_ptr<std::map<std::string, std::any>> getCurrentSettings() const;

 protected:                 
  SourceSettings();                                                                                                                                    
  /// Contains all settings that are about to be set using the update() function. 
  /// If update() is called these settings will be used to call all the processing 
  /// steps. When finished, all set values will be written into mCurrentSettings
  /// and mUpdateSettings gets reset.
  std::shared_ptr<std::map<std::string, std::any>> mUpdateSettings;
  /// Contains all settings currently set and playing
  std::shared_ptr<std::map<std::string, std::any>> mCurrentSettings;
};

} // namespace cs::audio

#endif // CS_AUDIO_SOURCE_SETTINGS_HPP
