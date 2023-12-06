////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "StreamingSource.hpp"
#include "internal/AlErrorHandling.hpp"
#include "internal/BufferManager.hpp"
#include "internal/FileReader.hpp"
#include "internal/SettingsMixer.hpp"
#include "logger.hpp"

#include <AL/al.h>
#include <AL/alext.h>
#include <any>
#include <filesystem>
#include <map>

namespace cs::audio {

StreamingSource::StreamingSource(std::string file, int bufferLength, int queueSize,
    std::shared_ptr<UpdateInstructor> UpdateInstructor)
    : SourceBase(file, UpdateInstructor)
    , mBuffers(std::vector<ALuint>(queueSize))
    , mAudioContainer(FileReader::AudioContainerStreaming())
    , mBufferLength(std::move(bufferLength))
    , mRefillBuffer(true)
    , mNotPlaying(true) {

  mAudioContainer.bufferLength = mBufferLength;

  alGetError(); // clear error code

  // create buffers
  alGenBuffers((ALsizei)mBuffers.size(), mBuffers.data());
  if (AlErrorHandling::errorOccurred()) {
    logger().warn("Failed to generate buffers!");
    return;
  }

  startStream();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

StreamingSource::StreamingSource()
    : SourceBase() {
}

////////////////////////////////////////////////////////////////////////////////////////////////////

StreamingSource::~StreamingSource() {
  if (mIsLeader) {
    alSourceStop(mOpenAlId);
    alSourceUnqueueBuffers(mOpenAlId, (ALsizei)mBuffers.size(), mBuffers.data());
    alDeleteBuffers((ALsizei)mBuffers.size(), mBuffers.data());
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool StreamingSource::updateStream() {
  if (!mIsLeader) {
    return true;
  }

  // possible improvement: instead of checking for playback and looping
  // in each frame, override the SourceSettings::set() function to also
  // set a state within the StreamingSource describing the playback and looping state

  // update the stream only if the source is supposed to be playing
  auto search = mPlaybackSettings->find("playback");
  if (search == mPlaybackSettings->end() || search->second.type() != typeid(std::string) ||
      std::any_cast<std::string>(search->second) != "play") {
    mNotPlaying = true;
    return false;
  }

  // get looping setting
  auto searchLooping = mPlaybackSettings->find("looping");
  if (searchLooping != mPlaybackSettings->end() && searchLooping->second.type() == typeid(bool)) {
    mAudioContainer.isLooping = std::any_cast<bool>(searchLooping->second);
  }

  if (mNotPlaying) {
    mRefillBuffer = true;
  } // source was just set to playing
  mNotPlaying         = false;
  bool updateRequired = false;

  ALint numBufferProcessed, state;
  alGetSourcei(mOpenAlId, AL_BUFFERS_PROCESSED, &numBufferProcessed);

  while (numBufferProcessed > 0) {

    ALuint bufferId;
    alSourceUnqueueBuffers(mOpenAlId, 1, &bufferId);
    if (AlErrorHandling::errorOccurred()) {
      logger().warn("Failed to unqueue buffer!");
      return false;
      ;
    }

    if (mRefillBuffer) {
      if (!FileReader::getNextStreamBlock(mAudioContainer)) {
        mRefillBuffer  = false;
        updateRequired = true;
        stop();
        numBufferProcessed--;
        continue;
      }
      fillBuffer(bufferId);

      alSourceQueueBuffers(mOpenAlId, 1, &bufferId);
      if (AlErrorHandling::errorOccurred()) {
        logger().warn("Failed to requeue buffer!");
        return false;
      }
    }
    numBufferProcessed--;
  }

  // restart source if underrun occurred
  alGetSourcei(mOpenAlId, AL_SOURCE_STATE, &state);
  if (state != AL_PLAYING) {
    alSourcePlay(mOpenAlId);
    if (AlErrorHandling::errorOccurred()) {
      logger().warn("Failed to restart playback of streaming source!");
      return false;
    }
  }

  return updateRequired;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool StreamingSource::setFile(std::string file) {
  if (!mIsLeader) {
    return true;
  }
  alGetError(); // clear error code

  // stop source if source is currently playing
  bool isPlaying = false;
  auto search    = mPlaybackSettings->find("playback");
  if (search != mPlaybackSettings->end() && search->second.type() == typeid(std::string) &&
      std::any_cast<std::string>(search->second) == "play") {

    isPlaying = true;
    alSourceStop(mOpenAlId);
    if (AlErrorHandling::errorOccurred()) {
      logger().warn("Failed to stop source!");
      return false;
    }
  }

  // remove current buffers
  ALuint buffers;
  alSourceUnqueueBuffers(mOpenAlId, (ALsizei)mBuffers.size(), &buffers);
  if (AlErrorHandling::errorOccurred()) {
    logger().warn("Failed to unqueue buffers!");
  }

  mFile = file;

  if (!startStream()) {
    return false;
  }

  if (isPlaying) {
    alSourcePlay(mOpenAlId);
    if (AlErrorHandling::errorOccurred()) {
      logger().warn("Failed to restart source!");
      return false;
    }
  }
  return true;
}

bool StreamingSource::startStream() {
  // check if file exists
  if (!std::filesystem::exists(mFile)) {
    logger().warn("{} file does not exist! Unable to fill buffer!", mFile);
    return false;
  }

  if (!FileReader::openStream(mFile, mAudioContainer)) {
    logger().warn("Failed to open stream for: {}!", mFile);
    return false;
  }

  // fill buffer
  for (auto buffer : mBuffers) {
    FileReader::getNextStreamBlock(mAudioContainer);
    if (mAudioContainer.splblockalign > 1) {
      alBufferi(buffer, AL_UNPACK_BLOCK_ALIGNMENT_SOFT, mAudioContainer.splblockalign);
    }
    fillBuffer(buffer);
  }

  if (AlErrorHandling::errorOccurred()) {
    logger().warn("Failed the inital stream buffering for: {}", mFile);
    return false;
  }

  // queue buffer
  alSourceQueueBuffers(mOpenAlId, (ALsizei)mBuffers.size(), mBuffers.data());

  if (AlErrorHandling::errorOccurred()) {
    logger().warn("Failed to queue the stream buffers for: {}", mFile);
    return false;
  }

  return true;
}

void StreamingSource::fillBuffer(ALuint buffer) {
  switch (mAudioContainer.formatType) {
  case FileReader::FormatType::Int16:
    alBufferData(buffer, mAudioContainer.format,
        std::get<std::vector<short>>(mAudioContainer.audioData).data(),
        (ALsizei)mAudioContainer.bufferSize, mAudioContainer.sfInfo.samplerate);
    break;

  case FileReader::FormatType::Float:
    alBufferData(buffer, mAudioContainer.format,
        std::get<std::vector<float>>(mAudioContainer.audioData).data(),
        (ALsizei)mAudioContainer.bufferSize, mAudioContainer.sfInfo.samplerate);
    break;

  default:
    alBufferData(buffer, mAudioContainer.format,
        std::get<std::vector<int>>(mAudioContainer.audioData).data(),
        (ALsizei)mAudioContainer.bufferSize, mAudioContainer.sfInfo.samplerate);
  }
  if (AlErrorHandling::errorOccurred()) {
    logger().warn("Failed to fill buffer for: {}...", mFile);
    mAudioContainer.print();
    return;
  }
}

} // namespace cs::audio