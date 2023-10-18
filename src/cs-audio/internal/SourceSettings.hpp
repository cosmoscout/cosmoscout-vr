////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CS_AUDIO_SOURCE_SETTINGS_HPP
#define CS_AUDIO_SOURCE_SETTINGS_HPP

#include "cs_audio_export.hpp"
// #include "UpdateBuilder.hpp"

#include <map>
#include <any>
#include <string>
#include <memory>

namespace cs::audio {

class UpdateBuilder;

class CS_AUDIO_EXPORT SourceSettings {
 public:
  /// @brief Sets a value in mUpdateSettings
  /// @param key setting type 
  /// @param value setting value 
  void set(std::string key, std::any value);

  /// @brief Returns the currently set settings 
  /// @return Pointer to the settings map
  std::shared_ptr<std::map<std::string, std::any>> getCurrentSettings() const;

  /// @brief Removes a key from the current and update settings.
  /// @param key key to remove
  void remove(std::string key);

  /// @brief Removes a key from the update settings.
  /// @param key key to remove
  void removeUpdate(std::string key);

 protected:                 
  SourceSettings(std::shared_ptr<UpdateBuilder> updateBuilder);                       
  /// Later assignment of updateBuilder needed because the audioController, which initializes the 
  /// updateBuilder, needs to initialize SourceSettings first.                                                                                                              
  SourceSettings();
  void setUpdateBuilder(std::shared_ptr<UpdateBuilder> updateBuilder);                                                                                                                                    
  /// Contains all settings that are about to be set using the update() function. 
  /// If update() is called these settings will be used to call all the processing 
  /// steps. When finished, all set values will be written into mCurrentSettings
  /// and mUpdateSettings gets reset.
  std::shared_ptr<std::map<std::string, std::any>> mUpdateSettings;
  /// Contains all settings currently set and playing.
  std::shared_ptr<std::map<std::string, std::any>> mCurrentSettings;
  /// UpdateBuilder to call to add Source/Group/Plugin to updateList 
  std::shared_ptr<UpdateBuilder>                   mUpdateBuilder;

  virtual void addToUpdateList() = 0;
};

} // namespace cs::audio

#endif // CS_AUDIO_SOURCE_SETTINGS_HPP
