////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CS_AUDIO_FILE_READER_HPP
#define CS_AUDIO_FILE_READER_HPP

#include "cs_audio_export.hpp"
#include <fstream>


namespace cs::audio {

struct AudioContainer {
  unsigned int format; 
  int sampleRate;
  int size;
  int splblockalign;
  std::variant<
   std::vector<short>, 
   std::vector<int>, 
   std::vector<float>> audioData;

  void print() {
    std::cout << "----WavContainer Info----" << std::endl;
    std::cout << "format: " << format << std::endl;
    std::cout << "sampleRate: " << sampleRate << "hz" << std::endl;
    std::cout << "size: " << size << std::endl;
    std::cout << "splblockalign: " << splblockalign << std::endl;
    std::cout << "-------------------------" << std::endl;
  }
};

struct AudioContainerStreaming : public AudioContainer {
  int bufferCounter = -1;
  int bufferSize; // size of 
  int currentBufferSize;
  std::ifstream in;

  void print() {
    std::cout << "----WavContainer Info----" << std::endl;
    std::cout << "format: " << format << std::endl;
    std::cout << "sampleRate: " << sampleRate << "hz" << std::endl;
    std::cout << "size: " << size << std::endl;
    std::cout << "splblockalign: " << splblockalign << std::endl;
    std::cout << "bufferCounter: " << bufferCounter << std::endl;
    std::cout << "bufferSize: " << bufferSize << std::endl;
    std::cout << "-------------------------" << std::endl;
  }

  ~AudioContainerStreaming() {
    in.close();
  }

  void reset() {
    bufferCounter = -1;
    bufferSize = 0;
    currentBufferSize = 0;
    format = 0; 
    sampleRate = 0;
    size = 0;
    in.close();
    // audioData = std::variant<
    //     std::vector<short>, 
    //     std::vector<int>, 
    //     std::vector<float>>();
  }
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
  static bool loadWAV(std::string fileName, AudioContainer& audioContainer);

  static const char* FormatName(ALenum format);
  
 private:
  enum FormatType {
    Int16,
    Float,
    IMA4,
    MSADPCM
  };
};

} // namespace cs::audio

#endif // CS_AUDIO_FILE_READER_HPP