////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CS_AUDIO_BASE_SOURCE_HPP
#define CS_AUDIO_BASE_SOURCE_HPP

#include "SourceSettings.hpp"
#include "UpdateInstructor.hpp"
#include "cs_audio_export.hpp"
#include <AL/al.h>
#include <any>
#include <map>

namespace cs::audio {

// forward declaration
class SourceGroup;

/// @brief This class implements the basic common functions of sources and is the parent for
/// both specific source types: streaming source and non-streaming sources.
class CS_AUDIO_EXPORT SourceBase : public SourceSettings,
                                   public std::enable_shared_from_this<SourceBase> {

 public:
  /// @brief This is the standard constructor used for non-cluster mode and cluster mode leader
  /// calls
  SourceBase(std::string file, std::shared_ptr<UpdateInstructor> UpdateInstructor);
  /// @brief This Constructor will create a dummy SourceBase which is used when a member of a
  /// cluster tries to create a SourceBase. Doing this will disable any functionality of this class.
  SourceBase();
  virtual ~SourceBase();

  /// @brief Sets setting to start playback. This call does not change the playback immediately.
  /// It still requires a call to AudioController::update().
  void play();

  /// @brief Sets setting to stop playback. This call does not change the playback immediately.
  /// It still requires a call to AudioController::update().
  void stop();

  /// @brief Sets setting to pause playback. This call does not change the playback immediately.
  /// It still requires a call to AudioController::update().
  void pause();

  /// @brief Virtual function to handle the change of file that is being played by a source.
  /// @param file Filepath to the new file.
  /// @return True if successful
  virtual bool setFile(std::string file) = 0;

  /// @return Returns the current file that is being played by the source.
  const std::string getFile() const;

  /// @return Returns the OpenAL ID
  const ALuint getOpenAlId() const;

  /// @brief Returns the current group
  /// @return Assigned group or nullptr if not part of any group
  const std::shared_ptr<SourceGroup> getGroup();

  /// @brief Assigns the source to a new group
  /// @param newGroup group to join
  void setGroup(std::shared_ptr<SourceGroup> newGroup);

  /// @brief leaves the current group
  void leaveGroup();

  /// @return Returns all settings (Source + Group + Controller) currently set and playing.
  const std::shared_ptr<const std::map<std::string, std::any>> getPlaybackSettings() const;

  // Is friend because the UpdateConstructor needs write permissions to the mPlaybackSettings.
  friend class UpdateConstructor;

 protected:
  /// OpenAL ID of source
  ALuint mOpenAlId;
  /// Currently set file to play
  std::string mFile;
  /// Ptr to the group that the source is part of
  std::weak_ptr<SourceGroup> mGroup;
  /// Contains all settings (Source + Group + Controller) currently set and playing.
  std::shared_ptr<std::map<std::string, std::any>> mPlaybackSettings;

  /// @brief Registers itself to the updateInstructor to be updated
  void addToUpdateList() override;
  /// @brief Deregisters itself from the updateInstructor
  void removeFromUpdateList() override;
};

} // namespace cs::audio

#endif // CS_AUDIO_BASE_SOURCE_HPP
