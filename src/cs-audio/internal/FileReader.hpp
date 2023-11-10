////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CS_AUDIO_FILE_READER_HPP
#define CS_AUDIO_FILE_READER_HPP

#include "cs_audio_export.hpp"


namespace cs::audio {

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

struct WavContainerStreaming : public WavContainer {
  int bufferCounter = -1;
  int bufferSize;
  int currentBuffer = 0;
};

class CS_AUDIO_EXPORT FileReader {
 public:
  FileReader(const FileReader& obj) = delete;
  FileReader(FileReader&&) = delete;

  FileReader& operator=(const FileReader&) = delete;
  FileReader& operator=(FileReader&&) = delete;

  /// @brief Reads the content of a .wav file and writes all the important information for OpenAL
  /// into the wavContainer.
  /// @param fileName path to the file to read
  /// @param wavContainer wavContainer to write into
  /// @return Whether the provided file path is a valid .wav file 
  static bool loadWAV(std::string fileName, WavContainer& wavContainer);
  static bool loadWAVPartially(std::string fileName, WavContainerStreaming& wavContainer);
  
 private:
  /// @brief Converts data in buffer up to the provided length to and int value
  /// @return int value
  static int convertToInt(char* buffer, int len);
  /// @brief Checks if the system is big or little endian
  /// @return True if big endian
  static bool isBigEndian();

  static bool readWAVHeader(std::string fileName, WavContainer& wavContainer);

  static std::vector<float> castToFloat(std::vector<char> input);
};

} // namespace cs::audio

#endif // CS_AUDIO_FILE_READER_HPP