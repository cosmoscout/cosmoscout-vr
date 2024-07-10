////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CS_AUDIO_SOURCE_SETTINGS_HPP
#define CS_AUDIO_SOURCE_SETTINGS_HPP

#include "UpdateConstructor.hpp"
#include "cs_audio_export.hpp"
#include <any>
#include <map>
#include <memory>
#include <string>

namespace cs::audio {

class UpdateInstructor;

/// @brief This class implements everything that is needed to define some properties for a source.
/// This property defining function is not limited to the source itself but also for a sourceGroup
/// and an audioController, which can both define properties for their sources.
class CS_AUDIO_EXPORT SourceSettings {
 public:
  virtual ~SourceSettings();

  /// @brief Sets a value in mUpdateSettings
  /// @param key setting type
  /// @param value setting value
  void set(std::string key, std::any value);

  /// @brief Returns the currently set settings for the sourceSettings instance.
  /// To get all settings currently playing on a source call Source::getPlaybackSettings().
  /// @return Pointer to the settings map
  const std::shared_ptr<const std::map<std::string, std::any>> getCurrentSettings() const;

  /// @brief Returns the currently set update settings for the sourceSettings instance.
  /// @return Pointer to the settings map
  const std::shared_ptr<const std::map<std::string, std::any>> getUpdateSettings() const;

  /// @brief Remove a setting and reset it to the default value.
  /// @param key Setting to remove.
  void remove(std::string key);

  /// @brief Removes a key from the update settings.
  /// @param key key to remove
  void removeUpdate(std::string key);

  // Is friend, because the UpdateConstructor needs write permissions to
  // mUpdateSettings and mCurrentSettings.
  friend class UpdateConstructor;

 protected:
  /// @brief This is a standard constructor used for non-cluster mode and cluster mode leader calls
  explicit SourceSettings(std::shared_ptr<UpdateInstructor> UpdateInstructor);
  /// @brief This is a standard constructor used for non-cluster mode and cluster mode leader calls
  explicit SourceSettings();
  /// @brief This Constructor will create a dummy SourceSetting which is used when a member of a
  /// cluster tries to create a SourceSetting. Doing this will disable any functionality of this
  /// class.
  explicit SourceSettings(bool isLeader);

  /// Later assignment of UpdateInstructor needed because the audioController, which initializes the
  /// UpdateInstructor, needs to initialize SourceSettings first.
  void setUpdateInstructor(std::shared_ptr<UpdateInstructor> UpdateInstructor);

  bool mIsLeader;
  /// Contains all settings that are about to be set using the AudioController::update() function.
  /// If update() is called these settings will be used to apply to a source. Not all settings might
  /// be set as they can be be overwritten by other settings higher up in the hierarchy (take a look
  /// at the UpdateConstructor for more details on this). After the update all set values will be
  /// written into mCurrentSettings and mUpdateSettings gets reset.
  std::shared_ptr<std::map<std::string, std::any>> mUpdateSettings;
  /// Contains all settings currently set by sourceSettings instance itself
  std::shared_ptr<std::map<std::string, std::any>> mCurrentSettings;
  /// UpdateInstructor to call to add sourceSettings instance to updateList
  std::shared_ptr<UpdateInstructor> mUpdateInstructor;

  /// @brief Add sourceSettings instance to the updateList. Each derived class needs to implement
  /// this by calling UpdateInstructor::update(shared_from_this())
  virtual void addToUpdateList() = 0;
  /// @brief Remove sourceSettings instance from the updateList. Each derived class needs to
  /// implement this by calling UpdateInstructor::removeUpdate(shared_from_this())
  virtual void removeFromUpdateList() = 0;
};

} // namespace cs::audio

#endif // CS_AUDIO_SOURCE_SETTINGS_HPP
