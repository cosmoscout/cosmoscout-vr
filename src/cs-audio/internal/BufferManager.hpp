////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CS_AUDIO_BUFFER_MANAGER_HPP
#define CS_AUDIO_BUFFER_MANAGER_HPP

#include "cs_audio_export.hpp"
#include <string>
#include <vector>
#include <AL/al.h>
#include <iostream>
#include <memory>

namespace cs::audio {

struct Buffer {
  std::string mFile;
  int         mUsageNumber;
  ALuint      mOpenAlId;

  Buffer(std::string file, ALuint openAlId) 
    : mFile(std::move(file))
    , mOpenAlId(std::move(openAlId)) {
    mUsageNumber = 1;  
  }
};

/// @brief This class handles the creation and deletion of buffers for non-streaming sources. 
/// This class should only be instantiated once.
class CS_AUDIO_EXPORT BufferManager {
 public:
  BufferManager(const BufferManager& obj) = delete;
  BufferManager(BufferManager&&) = delete;

  BufferManager& operator=(const BufferManager&) = delete;
  BufferManager& operator=(BufferManager&&) = delete;

  BufferManager();
  ~BufferManager();

  /// @brief Returns a buffer id containing the data for the provided file path.
  /// The BufferManager will check if a buffer for this file already exists and if so reuse the 
  /// the existing one. 
  /// @return Pair of bool and potential buffer id. Bool is false if an error occurred, which 
  /// means the buffer id is not valid.
  std::pair<bool, ALuint> getBuffer(std::string file);

  /// @brief Signals to the BufferManager that a Source is no longer using a buffer to the 
  /// provided file. If there are no more Sources using a buffer to a specific file, the 
  /// BufferManager will automatically delete the buffer.
  void removeBuffer(std::string file);
  
 private:
  /// @brief List of all current buffers
  std::vector<std::shared_ptr<Buffer>> mBufferList;
  
  /// @brief Creates a new Buffer if none already exists for the provided file path.
  std::pair<bool, ALuint> createBuffer(std::string file);
  
  /// @brief Deletes a buffer if it is no longer used by any Source.
  void deleteBuffer(std::vector<std::shared_ptr<Buffer>>::iterator bufferIt);
};

} // namespace cs::audio

#endif // CS_AUDIO_BUFFER_MANAGER_HPP