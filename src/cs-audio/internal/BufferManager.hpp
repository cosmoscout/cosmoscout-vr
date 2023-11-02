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
#include <variant>
#include <AL/al.h>
#include <iostream>

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

struct WavContainer {
  unsigned int format;
  int numberChannels;
  int sampleRate;
  int bitsPerSample;
  int size;
  std::variant<std::vector<char>, std::vector<float>> pcm; // actual audio data
  // std::vector<char> pcm;

  void print() {

    std::cout << "----WavContainer Info----" << std::endl;
    std::cout << "format: " << format << std::endl;
    std::cout << "numberChannels: " << numberChannels << std::endl;
    std::cout << "sampleRate: " << sampleRate << std::endl;
    std::cout << "bitsPerSample: " << bitsPerSample << std::endl;
    std::cout << "size: " << size << std::endl;
    std::cout << "type: " << (std::holds_alternative<std::vector<char>>(pcm) ? "char" : "float") << std::endl;
    std::cout << "-------------------------" << std::endl;
  }
};

class CS_AUDIO_EXPORT BufferManager {
 public:
  BufferManager(const BufferManager& obj) = delete;
  BufferManager(BufferManager&&) = delete;

  BufferManager& operator=(const BufferManager&) = delete;
  BufferManager& operator=(BufferManager&&) = delete;

  static std::shared_ptr<BufferManager> createBufferManager();
  ~BufferManager();

  /// @brief Returns an OpenAL id to a buffer containing the data for the provided file path.
  /// The BufferManager will check if a buffer for this file already exists and if so reuse the 
  /// the existing one. 
  /// @return Pair of bool and potential OpenAL id. Bool is false if an error occurred, which means the 
  /// OpenAL id is not a valid buffer.
  std::pair<bool, ALuint> getBuffer(std::string file);
  /// @brief Signals to the BufferManager that a Source is no longer using a buffer to the provided file.
  /// If there are no more Sources using a buffer to a specific file the BufferManager will automatically delete
  /// the buffer.
  void removeBuffer(std::string file);
  
 private:
  /// @brief List of all current buffers with 
  std::vector<std::shared_ptr<Buffer>> mBufferList;
  
  BufferManager();
  /// @brief Creates a new Buffer if none already exists for the provided file path.
  std::pair<bool, ALuint> createBuffer(std::string file);
  /// @brief Deletes a buffer if it is no longer used by any Source.
  void deleteBuffer(std::shared_ptr<Buffer> buffer);
};

} // namespace cs::audio

#endif // CS_AUDIO_BUFFER_MANAGER_HPP