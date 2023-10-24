////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "Source.hpp"
#include "internal/BufferManager.hpp"
#include "internal/alErrorHandling.hpp"
#include "internal/ProcessingStepsManager.hpp"
#include "internal/SettingsMixer.hpp"

#include <AL/al.h>
#include <map>
#include <any>

namespace cs::audio {

Source::Source(std::shared_ptr<BufferManager> bufferManager, 
  std::shared_ptr<ProcessingStepsManager> processingStepsManager,
  std::string file, std::shared_ptr<UpdateBuilder> updateBuilder)
  : SourceSettings(updateBuilder) 
  , mFile(std::move(file)) 
  , mBufferManager(std::move(bufferManager)) 
  , mProcessingStepsManager(std::move(processingStepsManager)) {
  
  alGetError(); // clear error code

  // TODO: check if file actually exists

  // generate new source  
  alGenSources((ALuint)1, &mOpenAlId);
  if (alErrorHandling::errorOccurred()) {
    logger().warn("Failed to generate OpenAL-Soft Source!");
    return;
  }

  // get buffer and bind buffer to source
  std::pair<bool, ALuint> buffer = mBufferManager->getBuffer(mFile);
  if (!buffer.first) {
    return;
  }
  alSourcei(mOpenAlId, AL_BUFFER, buffer.second);
  if (alErrorHandling::errorOccurred()) {
    logger().warn("Failed to bind buffer to source!");
    return;
  }
  // TODO: call process() with group and plugin settings
  // mProcessingStepsManager->process(mOpenAlId, mAudioControllerId, mCurrentSettings);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

Source::~Source() {
  alGetError(); // clear error code
  alDeleteSources(1, &mOpenAlId);
  if (alErrorHandling::errorOccurred()) {
    logger().warn("Failed to delete source!");
  }
  mBufferManager->removeBuffer(mFile);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool Source::play() const {
  alSourcePlay(mOpenAlId);
  if (alErrorHandling::errorOccurred()) {
    logger().warn("Failed to start playback of source!");
    return false;
  }
  return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool Source::stop() const {
  alSourceStop(mOpenAlId);
  if (alErrorHandling::errorOccurred()) {
    logger().warn("Failed to stop playback of source!");
    return false;
  }
  return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool Source::setFile(std::string file) {
  alGetError(); // clear error code
  // remove current buffer
  alSourcei(mOpenAlId, AL_BUFFER, NULL);
  if (alErrorHandling::errorOccurred()) {
    logger().warn("Failed to remove buffer from source!");
    return false;
  }
  mBufferManager->removeBuffer(mFile);
  
  // TODO: check if file exists
  
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
  return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::string Source::getFile() const {
  return mFile;   
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Source::addToUpdateList() {
  mUpdateBuilder->update(this);
}

} // namespace cs::audio