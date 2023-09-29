////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "BufferManager.hpp"
#include "FileReader.hpp"
#include "../logger.hpp"

#include <AL/al.h>

namespace cs::audio {

BufferManager::~BufferManager() {
  // delete all buffers
  // gather all buffer Ids to delete them in a single OpenAL call
  std::unique_ptr<ALuint[]> bufferIds(new ALuint[mBufferList.size()]);
  int i = 0;
  for (std::shared_ptr<Buffer> buffer : mBufferList) {
    bufferIds[i] = buffer->mOpenAlId;
    i++;   
  }
  alDeleteBuffers((ALsizei) mBufferList.size(), bufferIds.get());
  // TODO: Error handling
}

////////////////////////////////////////////////////////////////////////////////////////////////////

ALuint BufferManager::getBuffer(std::string file) {
  for (std::shared_ptr<Buffer> buffer : mBufferList) {
    if (buffer->mFile == file) {
      buffer->mUsageNumber++;
      return buffer->mOpenAlId;
    }
  }
  return createBuffer(file);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
 
ALuint BufferManager::createBuffer(std::string file) {
  alGetError(); // pop error stack

  // TODO: ist ein array wirklich nötig?  
  std::unique_ptr<ALuint> newBufferId;
  std::unique_ptr<ALuint[]> newBufferId2(new ALuint[1]);
  alGenBuffers(1, newBufferId2.get());
  // TODO: Error handling

  // read wave file and load into buffer
  unsigned int format;
	int channel, sampleRate, bps, size;	
	char* data = FileReader::loadWAV(file.c_str(), channel, sampleRate, bps, size, format);
	alBufferData(newBufferId2[0], format, data, size, sampleRate);
	delete[] data;
  // TODO: Error handling

  // add Buffer 
  mBufferList.push_back(std::make_shared<Buffer>(file, newBufferId2[0]));

  int bufferSize;
  alGetBufferi(newBufferId2[0], AL_SIZE, &bufferSize);
  std::cout << "size: " << bufferSize << std::endl;

  return newBufferId2[0];
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void BufferManager::removeBuffer(std::string file) {
  for (std::shared_ptr<Buffer> buffer : mBufferList) {
    if (buffer->mFile == file) {
      if (--buffer->mUsageNumber == 0) {
        deleteBuffer(buffer);
      }
      break;
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void BufferManager::deleteBuffer(std::shared_ptr<Buffer> bufferToDelete) {
  alGetError(); // pop error stack
  
  // delete buffer in OpenAL
  alDeleteBuffers((ALuint) 1, &bufferToDelete->mOpenAlId);
  // TODO: Error handling

  // delete buffer from bufferList
  int counter = 0;
  for (std::shared_ptr<Buffer> buffer : mBufferList) {
    if (buffer == bufferToDelete) {
       mBufferList.erase(mBufferList.begin() + counter);   
       break;
    }
    counter++;
  }
}

} // namespace cs::audio