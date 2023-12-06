////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CS_AUDIO_FILE_READER_HPP
#define CS_AUDIO_FILE_READER_HPP

#include "cs_audio_export.hpp"
#include <sndfile.h>
#include <variant>
#include <iostream>
#include <vector>
#include <AL/al.h>

namespace cs::audio {


class CS_AUDIO_EXPORT FileReader {
 public:
  enum FormatType {
    Int16,
    Float,
    IMA4,
    MSADPCM
  };

  struct AudioContainer {
    unsigned int format; 
    int size;
    int splblockalign;
    int byteblockalign;
    FormatType formatType;
    SF_INFO sfInfo;
    SNDFILE* sndFile;
    std::variant<
    std::vector<short>, 
    std::vector<int>, 
    std::vector<float>> audioData;

    void print() {
      std::cout << "----AudioContainer Info----" << std::endl;
      std::cout << "format: " << FileReader::getFormatName(formatType) << std::endl;
      std::cout << "sampleRate: " << sfInfo.samplerate << "hz" << std::endl;
      std::cout << "size: " << size << std::endl;
      std::cout << "splblockalign: " << splblockalign << std::endl;
      std::cout << "byteblockalign: " << byteblockalign << std::endl;
      std::cout << "formatType: " << formatType << std::endl;
    }

    void reset() {
      format = 0; 
      size = 0;
      splblockalign = 0;
      byteblockalign = 0;
      formatType = FormatType::Int16;
      sf_close(sndFile);

      if (std::holds_alternative<std::vector<short>>(audioData)) {
        std::get<std::vector<short>>(audioData).clear();
      } else if (std::holds_alternative<std::vector<float>>(audioData)) {
        std::get<std::vector<float>>(audioData).clear();
      } else {
        std::get<std::vector<int>>(audioData).clear();
      }    
    }
  };

  struct AudioContainerStreaming : public AudioContainer {
    int bufferCounter;
    int bufferLength; // in milliseconds 
    int blockCount;
    bool isLooping;
    sf_count_t bufferSize;

    void print() {
      AudioContainer::print();
      std::cout << "bufferCounter: " << bufferCounter << std::endl;
      std::cout << "blockCount: " << blockCount << std::endl;
      std::cout << "bufferSize: " << bufferSize << std::endl;
    }

    ~AudioContainerStreaming() {
      reset();
    }

    void reset() {
      AudioContainer::reset();
      bufferCounter = 0;
      bufferSize = 0;
      blockCount = 0;
      isLooping = false;
    }
  };
  FileReader(const FileReader& obj) = delete;
  FileReader(FileReader&&) = delete;

  FileReader& operator=(const FileReader&) = delete;
  FileReader& operator=(FileReader&&) = delete;

  /// @brief Reads the content of a .wav file and writes all the important information for OpenAL
  /// into the wavContainer.
  /// @param fileName path to file
  /// @param audioContainer audioContainer to write into
  /// @return Whether the provided file path is a valid .wav file 
  static bool loadFile(std::string fileName, AudioContainer& audioContainer);

  /// @return Name of audio format
  static const char* getFormatName(ALenum format);

  /// @brief Retrieves all necessary data from a file to start a stream
  /// @param fileName path to file
  /// @param audioContainer audioContainer to write into
  /// @return True if successful
  static bool openStream(std::string fileName, AudioContainerStreaming& audioContainer);

  /// @brief Retrieves the next chunk of data from a file to write into an OpenAL buffer
  /// @param audioContainer to write into
  /// @return False if source is not looping and no more buffers need to be filled
  /// to play the remaining content
  static bool getNextStreamBlock(AudioContainerStreaming& audioContainer);

 private:
  static bool readMetaData(std::string fileName, AudioContainer& audioContainer);
};

} // namespace cs::audio

#endif // CS_AUDIO_FILE_READER_HPP