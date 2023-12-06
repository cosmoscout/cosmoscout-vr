////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "Source.hpp"
#include "logger.hpp"
#include "internal/BufferManager.hpp"
#include "internal/alErrorHandling.hpp"
#include "internal/SettingsMixer.hpp"

#include <AL/al.h>
#include <map>
#include <filesystem>
#include <any>

namespace cs::audio {

Source::Source(std::shared_ptr<BufferManager> bufferManager, 
  std::string file, std::shared_ptr<UpdateInstructor> UpdateInstructor)
  : SourceBase(file, UpdateInstructor) 
  , mBufferManager(std::move(bufferManager)) {
  
  alGetError(); // clear error code

  // check if file exists
  if (!std::filesystem::exists(mFile)) {
    logger().warn("{} file does not exist! Unable to fill buffer!", mFile);
    return;
  }
  // get buffer
  std::pair<bool, ALuint> buffer = mBufferManager->getBuffer(mFile);
  if (!buffer.first) {
    return;
  }
  // bind buffer to source
  alSourcei(mOpenAlId, AL_BUFFER, buffer.second);
  if (alErrorHandling::errorOccurred()) {
    logger().warn("Failed to bind buffer to source!");
    return;
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

Source::Source()
  : SourceBase() {
}

////////////////////////////////////////////////////////////////////////////////////////////////////

Source::~Source() {
  if (mIsLeader) {
    alSourceStop(mOpenAlId);
    alSourcei(mOpenAlId, AL_BUFFER, 0);
    mBufferManager->removeBuffer(mFile);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool Source::setFile(std::string file) {
  if (!mIsLeader) { return true; }
  alGetError(); // clear error code

  ALint state;
  alGetSourcei(mOpenAlId, AL_SOURCE_STATE, &state);
  if (state == AL_PLAYING) {
    alSourceStop(mOpenAlId);
  }

  // remove current buffer
  alSourcei(mOpenAlId, AL_BUFFER, 0);
  if (alErrorHandling::errorOccurred()) {
    logger().warn("Failed to remove buffer from source!");
    return false;
  }
  mBufferManager->removeBuffer(mFile);
  
  // check if file exists
  if (!std::filesystem::exists(file)) {
    logger().warn("{} file does not exist! Unable to fill buffer!", file);
    return false;
  }
  mFile = file;
  
  // get buffer and bind buffer to source
  std::pair<bool, ALuint> buffer = mBufferManager->getBuffer(mFile);
  if (!buffer.first) {
    return false;
  }
  alSourcei(mOpenAlId, AL_BUFFER, buffer.second);
  if (alErrorHandling::errorOccurred()) {
    logger().warn("Failed to bind buffer to source!");
    return false;
  }

  if (state == AL_PLAYING) {
    alSourcePlay(mOpenAlId);
  }

  return true;
}

} // namespace cs::audio