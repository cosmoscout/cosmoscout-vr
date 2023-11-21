////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "FileReader.hpp"
#include "BufferManager.hpp"
#include "../logger.hpp"

#include <iostream>
#include <fstream>
#include <cstring>
#include <variant>
#include <algorithm>
#include <cstdlib>
#include <AL/al.h>
#include <AL/alext.h>

namespace cs::audio {

bool FileReader::loadWAV(std::string fileName, WavContainer& wavContainer)
{
  if (!readWAVHeader(fileName, wavContainer)) {
    return false;
  }
  std::ifstream in(fileName, std::ios::binary);
  // move reader to the data chunk
  in.seekg(44);

  if (wavContainer.bitsPerSample == 32) {
    auto charData = std::vector<char>(wavContainer.size);
    in.read(charData.data(), wavContainer.size); // data
    wavContainer.pcm = castToFloat(charData);

  } else {
    wavContainer.pcm = std::vector<char>(wavContainer.size);
    in.read(std::get<std::vector<char>>(wavContainer.pcm).data(), wavContainer.size); // data
  }

  return true;
}

bool FileReader::loadWAVPartially(std::string fileName, WavContainerStreaming& wavContainer)
{
  // Read wav header if this is the first buffer for the stream being read
  if (wavContainer.bufferCounter == -1) {
    if (!readWAVHeader(fileName, wavContainer)) {
      return false;
    }
    wavContainer.bufferCounter = 0;
    wavContainer.pcm = std::vector<char>(wavContainer.bufferSize / sizeof(char));
    wavContainer.in = std::ifstream(fileName, std::ios::binary);
    wavContainer.in.seekg(44); // move to the data chunk. '44' is the size of the WAV header
  }

  // Read the actual data from the file. If this buffer reaches the end of the file it will reset the 
  // buffer counter and the next buffer will start from the start of the file again.
  bool rewind = false;
  if ((wavContainer.bufferCounter + 1) * wavContainer.bufferSize >= wavContainer.size) {
    wavContainer.currentBufferSize = wavContainer.size - (wavContainer.bufferCounter * wavContainer.bufferSize);
    wavContainer.bufferCounter = 0;
    rewind = true;

  } else {
    wavContainer.currentBufferSize = wavContainer.bufferSize;
    wavContainer.bufferCounter++;
  }

  wavContainer.in.read(std::get<std::vector<char>>(wavContainer.pcm).data(), wavContainer.currentBufferSize);
  
  if (rewind) {
    wavContainer.in.seekg(44);
  }
  return true;
}

bool FileReader::readWAVHeader(std::string fileName, WavContainer& wavContainer) {
  char fileBuffer[4];
  std::ifstream in(fileName, std::ios::binary);

  // check if it is a valid wave file:
  in.read(fileBuffer, 4);
  if (strncmp(fileBuffer, "RIFF", 4) != 0) {
    return false;
  }

  in.read(fileBuffer, 4); // ChunkSize            -- RIFF chunk descriptor
  in.read(fileBuffer, 4); // Format
  in.read(fileBuffer, 4); // SubChunk 1 id        -- fmt sub-chunk
  in.read(fileBuffer, 4); // SubChunk 1 size
  in.read(fileBuffer, 2); // AudioFormat
  in.read(fileBuffer, 2); // Number Channels
  wavContainer.numberChannels = convertToInt(fileBuffer, 2);
  in.read(fileBuffer, 4); // Sample Rate
  wavContainer.sampleRate = convertToInt(fileBuffer, 4);
  in.read(fileBuffer, 4); // Byte Rate
  in.read(fileBuffer, 2); // Block Align
  in.read(fileBuffer, 2); // Bits per Sample
  wavContainer.bitsPerSample = convertToInt(fileBuffer, 2);
  in.read(fileBuffer, 4); // SubChunk 2 id        -- data sub-chunk
  in.read(fileBuffer, 4); // SubChunk 2 Size
  wavContainer.size = convertToInt(fileBuffer, 4);  

  // Mono
  if (wavContainer.numberChannels == 1) {
    switch (wavContainer.bitsPerSample) {
      case 8:
        wavContainer.format = AL_FORMAT_MONO8;
        break;
      case 16:
        wavContainer.format = AL_FORMAT_MONO16;
    }
  // Stereo
  } else {
    switch (wavContainer.bitsPerSample) {
      case 8:
        wavContainer.format = AL_FORMAT_STEREO8;
        break;
      case 16:
        wavContainer.format = AL_FORMAT_STEREO16;
    }
  }
  return true;
}

int FileReader::convertToInt(char* buffer, int len)
{
    static bool bigEndian = isBigEndian();
    int a = 0;
    if (!bigEndian)
        for (int i = 0; i < len; i++)
            ((char*)&a)[i] = buffer[i];
    else
        for (int i = 0; i < len; i++)
            ((char*)&a)[3 - i] = buffer[i];
    return a;
}

bool FileReader::isBigEndian()
{
    int a = 1;
    return !((char*)&a)[0];
}

std::vector<float> FileReader::castToFloat(std::vector<char> input)
{
  std::vector<float> output;
  for (char element : input) {
    output.push_back( (+element + 128) / 255.0f * 2.0f - 1.0f );
  }
  return output;
}


} // namespace cs::audio