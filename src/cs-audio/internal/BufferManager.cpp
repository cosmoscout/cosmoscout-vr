////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "BufferManager.hpp"

#include <AL/al.h>

namespace cs::audio {

BufferManager::~BufferManager() {
  // delete all buffers
  // gather all buffer Ids to delete them in a single OpenAL call
  std::unique_ptr<ALuint[]> bufferIds(new ALuint[bufferList.size()]);
  int i = 0;
  for (std::shared_ptr<Buffer> buffer : bufferList) {
    bufferIds[i] = buffer->openAlId;
    i++;   
  }
  alDeleteBuffers((ALuint) bufferList.size(), bufferIds.get());
  // TODO: Error handling
}

////////////////////////////////////////////////////////////////////////////////////////////////////

ALuint BufferManager::getBuffer(std::string file) {
  for (std::shared_ptr<Buffer> buffer : bufferList) {
    if (buffer->file == file) {
      buffer->usageNumber++;
      return buffer->openAlId;
    }
  }
  return createBuffer(file);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
 
ALuint BufferManager::createBuffer(std::string file) {
  alGetError(); // pop error stack

  std::unique_ptr<ALuint> newBufferId;
  alGenBuffers((ALuint) 1, newBufferId.get());
  // TODO: Error handling

  // TODO: read file and fill buffer with the data
  // alBufferData(*newBufferId, format, data, size, sampleRate);

  bufferList.push_back(std::make_shared<Buffer>(file, 1, *newBufferId));
  return *newBufferId;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void BufferManager::removeBuffer(std::string file) {
  for (std::shared_ptr<Buffer> buffer : bufferList) {
    if (buffer->file == file) {
      if (--buffer->usageNumber == 0) {
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
  alDeleteBuffers((ALuint) 1, &bufferToDelete->openAlId);
  // TODO: Error handling

  // delete buffer from bufferList
  int counter = 0;
  for (std::shared_ptr<Buffer> buffer : bufferList) {
    if (buffer == bufferToDelete) {
       bufferList.erase(bufferList.begin() + counter);   
       break;
    }
    counter++;
  }
}

} // namespace cs::audio