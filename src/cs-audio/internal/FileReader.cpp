////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "FileReader.hpp"
#include "BufferManager.hpp"

#include <iostream>
#include <fstream>
#include <cstring>
#include <variant>
#include <AL/al.h>
#include <AL/alext.h>

namespace cs::audio {

bool FileReader::loadWAV(std::string fileName, WavContainer& wavContainer)
{
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
  // wavContainer.pcm = std::vector<char>(wavContainer.size);
  // in.read(wavContainer.pcm.data(), wavContainer.size); // data

  // Mono
  if (wavContainer.numberChannels == 1) {
    switch (wavContainer.bitsPerSample) {
      case 8:
        wavContainer.format = AL_FORMAT_MONO8;
        break;
      case 16:
        wavContainer.format = AL_FORMAT_MONO16;
        break;
      case 32:
        wavContainer.format = AL_FORMAT_MONO_FLOAT32;
    }
  // Stereo
  } else {
    switch (wavContainer.bitsPerSample) {
      case 8:
        wavContainer.format = AL_FORMAT_STEREO8;
        break;
      case 16:
        wavContainer.format = AL_FORMAT_STEREO16;
        break;
      case 32:
        wavContainer.format = AL_FORMAT_STEREO_FLOAT32;
    }
  }
  if (wavContainer.bitsPerSample == 32) {
    wavContainer.pcm = std::vector<float>(wavContainer.size);
    in.read(reinterpret_cast<char*>(std::get<std::vector<float>>(wavContainer.pcm).data()), wavContainer.size); // data
  } else {
    wavContainer.pcm = std::vector<char>(wavContainer.size);
    in.read(std::get<std::vector<char>>(wavContainer.pcm).data(), wavContainer.size); // data
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

} // namespace cs::audio