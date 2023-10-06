////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "Source.hpp"
#include "../cs-core/AudioEngine.hpp"
#include "internal/BufferManager.hpp"
#include "internal/openAlError.hpp"

#include <AL/al.h>

// FadeIn FadeOut?

namespace cs::audio {

Source::Source(std::shared_ptr<BufferManager> bufferManager, std::string file, std::shared_ptr<SourceSettings> settings) 
  : mFile(std::move(file)) 
  , mBufferManager(std::move(bufferManager)) 
  , mSettings(std::move(settings)) {

  alGetError(); // clear error code

  // TODO: check if file actually exists

  // generate new source  
  alGenSources((ALuint)1, &mOpenAlId);
  if (alErrorHandling::errorOccurd()) {
    logger().warn("Failed to generate OpenAL-Soft Source!");
  }

  // get buffer and bind buffer to source
  alSourcei(mOpenAlId, AL_BUFFER, mBufferManager->getBuffer(mFile));
  if (alErrorHandling::errorOccurd()) {
    logger().warn("Failed to bind buffer to source!");
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

Source::~Source() {
  alGetError(); // clear error code
  alDeleteSources(1, &mOpenAlId);
  if (alErrorHandling::errorOccurd()) {
    logger().warn("Failed to delete source!");
  }
  mBufferManager->removeBuffer(mFile);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool Source::play() {
  alSourcePlay(mOpenAlId);
  if (alErrorHandling::errorOccurd()) {
    logger().warn("Failed to start playback of source!");
    return false;
  }
  return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool Source::stop() {
  alSourceStop(mOpenAlId);
  if (alErrorHandling::errorOccurd()) {
    logger().warn("Failed to stop playback of source!");
    return false;
  }
  return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Source::update() {
  // call all processing steps
  mProcessingStepsManager->process(mOpenAlId, settings);

  // write changed values into mCurrentSettings
  // TODO
    
  // reset settings
  settings = std::make_shared<SourceSettings>(); 
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool Source::setFile(std::string file) {
  alGetError(); // clear error code
  // alSourceStop(mOpenAlId);
  alSourcei(mOpenAlId, AL_BUFFER, NULL);
  if (alErrorHandling::errorOccurd()) {
    logger().warn("Failed to remove buffer from source!");
    return false;
  }
  mBufferManager->removeBuffer(mFile);
  
  // TODO: check if file exists
  
  mFile = file;
  mBufferManager->getBuffer(mFile);
  return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::string Source::getFile() const {
  return mFile;   
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::shared_ptr<SourceSettings> Source::getSettings() const {
  return mCurrentSettings;
}

} // namespace cs::audio