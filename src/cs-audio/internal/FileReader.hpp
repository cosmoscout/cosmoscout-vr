////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CS_AUDIO_FILE_READER_HPP
#define CS_AUDIO_FILE_READER_HPP

#include "cs_audio_export.hpp"


namespace cs::audio {

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
  
 private:
  /// @brief Converts data in buffer up to the provided length to and int value
  /// @return int value
  static int convertToInt(char* buffer, int len);
  /// @brief Checks if the system is big or little endian
  /// @return True if big endian
  static bool isBigEndian();

  static std::vector<float> castToFloat(std::vector<char> input);
};

} // namespace cs::audio

#endif // CS_AUDIO_FILE_READER_HPP