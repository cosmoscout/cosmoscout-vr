////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CS_AUDIO_SOURCE_HPP
#define CS_AUDIO_SOURCE_HPP

#include "cs_audio_export.hpp"
#include "internal/SourceSettings.hpp"
#include "internal/BufferManager.hpp"
#include "internal/ProcessingStepsManager.hpp"
#include "internal/UpdateInstructor.hpp"

#include <AL/al.h>
#include <map>
#include <any>

namespace cs::audio {

// forward declaration
class SourceGroup;

class CS_AUDIO_EXPORT Source 
  : public SourceSettings
  , public std::enable_shared_from_this<Source> {
    
 public:
  ~Source();
  
  /// @brief Sets setting to start playback
  void play();

  /// @brief Sets setting to stop playback
  void stop();

  /// @brief Sets setting to pause playback
  void pause();

  /// @brief Sets a new file to be played by the source.
  /// @return Whether it was successful
  bool setFile(std::string file); // TODO: what happens if the source is currently playing?

  /// @return Returns the current file that is getting played by the source.
  std::string getFile() const;

  /// @return Returns to OpenAL ID
  ALuint getOpenAlId() const;

  /// @return Returns all settings (Source + Group + Controller) currently set and playing.
  std::shared_ptr<std::map<std::string, std::any>> getPlaybackSettings() const;

  // TODO: Constructor in private ausprobieren

  Source(std::shared_ptr<BufferManager> bufferManager, 
  std::shared_ptr<ProcessingStepsManager> processingStepsManager,
  std::string file, std::shared_ptr<UpdateInstructor> UpdateInstructor);

  // friend class cs::core::AudioEngine;
  friend class SourceGroup;
  friend class UpdateConstructor;
    
 private:
  std::shared_ptr<BufferManager>                   mBufferManager;
  std::shared_ptr<ProcessingStepsManager>          mProcessingStepsManager;
  ALuint                                           mOpenAlId; 
  /// Currently set file to play
  std::string                                      mFile;
  /// Ptr to the group that the source is part of
  std::shared_ptr<SourceGroup>                     mGroup;
  /// Contains all settings (Source + Group + Controller) currently set and playing. 
  std::shared_ptr<std::map<std::string, std::any>> mPlaybackSettings;

  /// @brief registers itself to the updateInstructor to be updated 
  void addToUpdateList();
};

} // namespace cs::audio

#endif // CS_AUDIO_SOURCE_HPP
