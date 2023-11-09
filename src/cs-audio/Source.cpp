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
  std::string file, std::shared_ptr<UpdateInstructor> UpdateInstructor)
  : SourceSettings(UpdateInstructor) 
  , std::enable_shared_from_this<Source>()
  , mFile(std::move(file)) 
  , mBufferManager(std::move(bufferManager)) 
  , mProcessingStepsManager(std::move(processingStepsManager))
  , mPlaybackSettings(std::make_shared<std::map<std::string, std::any>>()) {
  
  alGetError(); // clear error code

  // generate new source  
  alGenSources((ALuint)1, &mOpenAlId);
  if (alErrorHandling::errorOccurred()) {
    logger().warn("Failed to generate OpenAL-Soft Source!");
    return;
  }
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
  
  // positions needs to be set relative in case the listener moves:
  // set position to listener relative
  alSourcei(mOpenAlId, AL_SOURCE_RELATIVE, AL_TRUE);
  if (alErrorHandling::errorOccurred()) {
    logger().warn("Failed to set source position specification to relative!");
    return;
  }

  alSource3i(mOpenAlId, AL_POSITION, 0, 0, 0);
  if (alErrorHandling::errorOccurred()) {
    logger().warn("Failed to set source position to (0, 0, 0)!");
    return;
  }
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

void Source::play() {
  set("playback", std::string("play"));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Source::stop() {
  set("playback", std::string("stop"));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Source::pause() {
  set("playback", std::string("pause"));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool Source::setFile(std::string file) {
  alGetError(); // clear error code

  ALint state;
  alGetSourcei(mOpenAlId, AL_SOURCE_STATE, &state);
  if (state == AL_PLAYING) {
    alSourceStop(mOpenAlId);
  }

  // remove current buffer
  alSourcei(mOpenAlId, AL_BUFFER, NULL);
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

////////////////////////////////////////////////////////////////////////////////////////////////////

std::string Source::getFile() const {
  return mFile;   
}

////////////////////////////////////////////////////////////////////////////////////////////////////

ALuint Source::getOpenAlId() const {
  return mOpenAlId;
}

std::shared_ptr<std::map<std::string, std::any>> Source::getPlaybackSettings() const {
  return mPlaybackSettings;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Source::addToUpdateList() {
  mUpdateInstructor->update(shared_from_this());
}

} // namespace cs::audio