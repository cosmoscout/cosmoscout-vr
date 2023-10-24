////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "BufferManager.hpp"
#include "FileReader.hpp"
#include "alErrorHandling.hpp"
#include "../logger.hpp"

#include <AL/al.h>
#include <iostream>

namespace cs::audio {

std::shared_ptr<BufferManager> BufferManager::createBufferManager() {
  static auto bufferManager = std::shared_ptr<BufferManager>(new BufferManager());
  return bufferManager;  
}

////////////////////////////////////////////////////////////////////////////////////////////////////

BufferManager::BufferManager()
  : mBufferList(std::vector<std::shared_ptr<Buffer>>()) {
}

////////////////////////////////////////////////////////////////////////////////////////////////////

BufferManager::~BufferManager() {
  alGetError(); // clear error code
  // delete all buffers
  // gather all buffer Ids to delete them in a single OpenAL call
  std::unique_ptr<ALuint[]> bufferIds(new ALuint[mBufferList.size()]);
  int i = 0;
  for (std::shared_ptr<Buffer> buffer : mBufferList) {
    bufferIds[i] = buffer->mOpenAlId;
    i++;   
  }
  alDeleteBuffers((ALsizei) mBufferList.size(), bufferIds.get());
  if (alErrorHandling::errorOccurred()) {
    logger().warn("Failed to delete (all) buffers!");
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::pair<bool, ALuint> BufferManager::getBuffer(std::string file) {
  for (std::shared_ptr<Buffer> buffer : mBufferList) {
    if (buffer->mFile == file) {
      buffer->mUsageNumber++;
      return std::make_pair(true, buffer->mOpenAlId);
    }
  }
  logger().debug(-1);
  auto x = createBuffer(file);
  logger().debug(2);
  return x;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::pair<bool, ALuint> BufferManager::createBuffer(std::string file) {
  alGetError(); // clear error code

  // create buffer
  ALuint newBufferId;
  alGenBuffers((ALsizei) 1, &newBufferId);
  if (alErrorHandling::errorOccurred()) {
    logger().warn("Failed to generate buffer!");
    return std::make_pair(false, newBufferId);
  }

  // read wave file
  WavContainer wavContainer;
  if (!FileReader::loadWAV(file.c_str(), wavContainer)) {
    logger().warn("{} is not a valid wave file! Unable to create buffer!", file);
    alDeleteBuffers((ALsizei) 1, &newBufferId);
    return std::make_pair(false, newBufferId);
  }

  // load wave into buffer
  alBufferData(newBufferId, wavContainer.format, wavContainer.data, 
    wavContainer.size, wavContainer.sampleRate);
  if (alErrorHandling::errorOccurred()) {
    logger().warn("Failed to fill buffer with data!");
    alDeleteBuffers((ALsizei) 1, &newBufferId);
    return std::make_pair(false, newBufferId);
  }

  // add Buffer 
  mBufferList.push_back(std::make_shared<Buffer>(file, newBufferId));

  return std::make_pair(true, newBufferId);
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
  alGetError(); // clear error code
  
  // delete buffer in OpenAL
  alDeleteBuffers((ALsizei) 1, &(bufferToDelete->mOpenAlId));
  if (alErrorHandling::errorOccurred()) {
    logger().warn("Failed to delete single buffer!");
  }

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