////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CS_AUDIO_BASE_SOURCE_HPP
#define CS_AUDIO_BASE_SOURCE_HPP

#include "cs_audio_export.hpp"
#include "SourceSettings.hpp"
#include "UpdateInstructor.hpp"

#include <AL/al.h>
#include <map>
#include <any>

namespace cs::audio {

// forward declaration
class SourceGroup;

class CS_AUDIO_EXPORT SourceBase 
  : public SourceSettings
  , public std::enable_shared_from_this<SourceBase> {
    
 public:
  ~SourceBase();
  
  /// @brief Sets setting to start playback
  void play();

  /// @brief Sets setting to stop playback
  void stop();

  /// @brief Sets setting to pause playback
  void pause();

  virtual bool setFile(std::string file) = 0;

  /// @return Returns the current file that is being played by the source.
  const std::string getFile() const;

  /// @return Returns the OpenAL ID
  const ALuint getOpenAlId() const;

  /// @brief Returns the current group
  /// @return Assigned group or nullptr if not part of any group
  const std::shared_ptr<SourceGroup> getGroup();

  /// @brief Assigned the source to a new group
  /// @param newGroup group to join
  void setGroup(std::shared_ptr<SourceGroup> newGroup);

  /// @brief leaves the current group
  void leaveGroup();

  /// @return Returns all settings (Source + Group + Controller) currently set and playing.
  const std::shared_ptr<std::map<std::string, std::any>> getPlaybackSettings() const;

  SourceBase(std::string file, std::shared_ptr<UpdateInstructor> UpdateInstructor);

  // friend class cs::core::AudioEngine;
  friend class SourceGroup;
  friend class UpdateConstructor;
    
 protected:
  ALuint                                           mOpenAlId; 
  /// Currently set file to play
  std::string                                      mFile;
  /// Ptr to the group that the source is part of
  std::weak_ptr<SourceGroup>                       mGroup;
  /// Contains all settings (Source + Group + Controller) currently set and playing. 
  std::shared_ptr<std::map<std::string, std::any>> mPlaybackSettings;

  /// @brief register itself to the updateInstructor to be updated 
  void addToUpdateList() override;
  /// @brief deregister itself from the updateInstructor 
  void removeFromUpdateList() override;
};

} // namespace cs::audio

#endif // CS_AUDIO_BASE_SOURCE_HPP
