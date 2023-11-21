////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "StreamingSource.hpp"
#include "logger.hpp"
#include "internal/BufferManager.hpp"
#include "internal/alErrorHandling.hpp"
#include "internal/SettingsMixer.hpp"
#include "internal/FileReader.hpp"

#include <AL/al.h>
#include <map>
#include <filesystem>
#include <any>

namespace cs::audio {

StreamingSource::StreamingSource(std::string file, int bufferSize, int queueSize,
  std::shared_ptr<UpdateInstructor> UpdateInstructor)
  : SourceBase(file, UpdateInstructor)
  , mBufferSize(std::move(bufferSize))
  , mBuffers(std::vector<ALuint>(queueSize)) 
  , mWavContainer(WavContainerStreaming()) { 

  alGetError(); // clear error code

  // get buffer
  alGenBuffers((ALsizei) mBuffers.size(), mBuffers.data());
  if (alErrorHandling::errorOccurred()) {
    logger().warn("Failed to generate buffers!");
    return;
  }

  // check if file exists
  if (!std::filesystem::exists(mFile)) {
    logger().warn("{} file does not exist! Unable to fill buffer!", mFile);
    return;
  }

  // fill buffer
  mWavContainer.bufferSize = mBufferSize;
  for (auto buffer : mBuffers) {

    if (!FileReader::loadWAVPartially(mFile, mWavContainer)) {
      logger().debug("Failed to loadWAVPartially");
    }

    alBufferData(buffer, mWavContainer.format, 
      std::get<std::vector<char>>(mWavContainer.pcm).data(), 
      mWavContainer.currentBufferSize, mWavContainer.sampleRate);
  }

  // queue buffer
  alSourceQueueBuffers(mOpenAlId, (ALsizei)mBuffers.size(), mBuffers.data());
}

////////////////////////////////////////////////////////////////////////////////////////////////////

StreamingSource::~StreamingSource() {
  alSourceStop(mOpenAlId);
  alSourceUnqueueBuffers(mOpenAlId, (ALsizei)mBuffers.size(), mBuffers.data());
  alDeleteBuffers((ALsizei) mBuffers.size(), mBuffers.data());
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void StreamingSource::updateStream() {
  // update the stream only if the source is supposed to be playing
  auto search = mPlaybackSettings->find("playback");
  if (search == mPlaybackSettings->end() || 
      search->second.type() != typeid(std::string) ||
      std::any_cast<std::string>(search->second) != "play") {
    return;
  }

  ALint numBufferProcessed, state;
  alGetSourcei(mOpenAlId, AL_BUFFERS_PROCESSED, &numBufferProcessed);

  while (numBufferProcessed > 0) {
    ALuint bufferId;
    alSourceUnqueueBuffers(mOpenAlId, 1, &bufferId);
    if (alErrorHandling::errorOccurred()) {
      logger().warn("Failed to unqueue buffer!");
      return;
    }
    
    FileReader::loadWAVPartially(mFile, mWavContainer);
  
    alBufferData(bufferId, mWavContainer.format, 
      std::get<std::vector<char>>(mWavContainer.pcm).data(), 
      mWavContainer.bufferSize, mWavContainer.sampleRate);
    if (alErrorHandling::errorOccurred()) {
      logger().warn("Failed to refill streaming buffer!");
      return;
    }

    alSourceQueueBuffers(mOpenAlId, 1, &bufferId);
    if (alErrorHandling::errorOccurred()) {
      logger().warn("Failed to requeue buffer!");
      return;
    }
    numBufferProcessed--;
  } 
  
  // restart source if underrun occurred
  alGetSourcei(mOpenAlId, AL_SOURCE_STATE, &state);
  if (state != AL_PLAYING) {
    alSourcePlay(mOpenAlId);
    if (alErrorHandling::errorOccurred()) {
      logger().warn("Failed to restart playback of streaming source!");
      return;
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool StreamingSource::setFile(std::string file) {
  alGetError(); // clear error code

  bool isPlaying = false;
  auto search = mPlaybackSettings->find("playback");
  if (search != mPlaybackSettings->end() && 
      search->second.type() == typeid(std::string) &&
      std::any_cast<std::string>(search->second) == "play") {
    
    isPlaying = true;
    alSourceStop(mOpenAlId);
    if (alErrorHandling::errorOccurred()) {
      logger().warn("Failed to stop source!");
      return false;
    }
  }

  // remove current buffers
  ALuint x; // TODO: do better
  alSourceUnqueueBuffers(mOpenAlId, (ALsizei)mBuffers.size(), &x);
  if (alErrorHandling::errorOccurred()) {
    logger().warn("Failed to unqueue buffers!");
  }
  
  // check if file exists
  if (!std::filesystem::exists(file)) {
    logger().warn("{} file does not exist! Unable to fill buffer!", file);
    return false;
  }
  mFile = file;

  mWavContainer.reset();
  mWavContainer.bufferSize = mBufferSize;

  // fill buffer
  for (auto buffer : mBuffers) {
    if (!FileReader::loadWAVPartially(mFile, mWavContainer)) {
      logger().debug("Failed to loadWAVPartially");
    }

    alBufferData(buffer, mWavContainer.format, 
      std::get<std::vector<char>>(mWavContainer.pcm).data(), 
      mWavContainer.currentBufferSize, mWavContainer.sampleRate);
  }

  // queue buffer
  alSourceQueueBuffers(mOpenAlId, (ALsizei)mBuffers.size(), mBuffers.data());

  if (isPlaying) {
    alSourcePlay(mOpenAlId);
    if (alErrorHandling::errorOccurred()) {
      logger().warn("Failed to restart source!");
      return false;
    }
  }

  return true;
}

} // namespace cs::audio