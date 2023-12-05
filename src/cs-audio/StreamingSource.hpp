////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CS_AUDIO_STREAMING_SOURCE_HPP
#define CS_AUDIO_STREAMING_SOURCE_HPP

#include "cs_audio_export.hpp"
#include "internal/SourceBase.hpp"
#include "internal/BufferManager.hpp"
#include "internal/UpdateInstructor.hpp"
#include "internal/FileReader.hpp"

#include <AL/al.h>
#include <map>
#include <any>

namespace cs::audio {

// forward declaration
class SourceGroup;

/// @brief This is the derived source class for streaming sources. This means that not the whole 
/// file is being read and written into the buffer but only small chunks at a time. StreamingSource
/// B´buffers cannot be shared among sources. Each StreamingSource handles it's buffers on it's own.
/// One current disadvantage is that StreamingSources can only be played after the loading screen 
/// because the CosmoScout update cycle is needed in order to update the changing buffers. 
class CS_AUDIO_EXPORT StreamingSource : public SourceBase {
    
 public:
  StreamingSource(std::string file, int bufferLength, int queueSize,
    std::shared_ptr<UpdateInstructor> UpdateInstructor);
  ~StreamingSource();

  /// @brief Sets a new file to be played by the source.
  /// @return true if successful
  bool setFile(std::string file) override;

  /// @brief Checks if any buffer finished playing and if so, requeues the buffer with new data. 
  /// @return True if a AudioController::update() is required. This is done to set the playback 
  /// state after the stream finished playing and looping is not enabled.
  bool updateStream();  

 private:
  /// @brief Fills an OpenAL buffer with already read data from a file
  /// @param buffer buffer to write to
  void fillBuffer(ALuint buffer);

  /// @brief Starts a new stream from a new file
  /// @return True if successful
  bool startStream();

  /// List of all OpenAL buffer IDs being used by the source
  std::vector<ALuint>     mBuffers;
  /// Contains all information regarding a file/buffer that is needed. 
  AudioContainerStreaming mAudioContainer;
  /// Length of each buffer in milliseconds
  int                     mBufferLength;
  /// Specifies whether buffers should still be filled in a stream update.
  /// Is false if no new buffer is required to play the remaining content.
  bool                    mRefillBuffer;
  /// Specifies whether the source was playing in the last frame
  bool                    mNotPlaying;
};

} // namespace cs::audio

#endif // CS_AUDIO_STREAMING_SOURCE_HPP