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

StreamingSource::StreamingSource(std::string file, 
  std::shared_ptr<UpdateInstructor> UpdateInstructor)
  : SourceBase(file, UpdateInstructor)
  , mBuffers(std::vector<ALuint>(2)) 
  , mWavContainer(WavContainerStreaming()){ 
  
  alGetError(); // clear error code

  // get buffer
  alGenBuffers((ALsizei) 2, mBuffers.data());
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
  mWavContainer.bufferSize = 10240;
  FileReader::loadWAVPartially(mFile, mWavContainer);
  
  alBufferData(mBuffers[0], mWavContainer.format, 
    std::get<std::vector<char>>(mWavContainer.pcm).data(), 
    mWavContainer.size, mWavContainer.sampleRate);

  FileReader::loadWAVPartially(mFile, mWavContainer);

  alBufferData(mBuffers[1], mWavContainer.format, 
    std::get<std::vector<char>>(mWavContainer.pcm).data(), 
    mWavContainer.size, mWavContainer.sampleRate);

  // queue buffer
  alSourceQueueBuffers(mOpenAlId, 2, mBuffers.data());
}

////////////////////////////////////////////////////////////////////////////////////////////////////

StreamingSource::~StreamingSource() {

}

////////////////////////////////////////////////////////////////////////////////////////////////////

void StreamingSource::updateStream() {
  logger().debug("update stream");
  ALint numBufferProcessed;
  alGetSourcei(mOpenAlId, AL_BUFFERS_PROCESSED, &numBufferProcessed);

  if (numBufferProcessed == 1) {
    alSourceUnqueueBuffers(mOpenAlId, 1, &(mBuffers[mWavContainer.currentBuffer]));
    
    FileReader::loadWAVPartially(mFile, mWavContainer);
  
    alBufferData(mBuffers[mWavContainer.currentBuffer], mWavContainer.format, 
      std::get<std::vector<char>>(mWavContainer.pcm).data(), 
      mWavContainer.size, mWavContainer.sampleRate);

    alSourceQueueBuffers(mOpenAlId, 1, &(mBuffers[mWavContainer.currentBuffer]));
    mWavContainer.currentBuffer = !mWavContainer.currentBuffer;
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool StreamingSource::setFile(std::string file) {
  alGetError(); // clear error code

  ALint state;
  alGetSourcei(mOpenAlId, AL_SOURCE_STATE, &state);
  if (state == AL_PLAYING) {
    alSourceStop(mOpenAlId);
  }

  // remove current buffer
  
  // check if file exists
  
  // get buffer and bind buffer to source


  if (state == AL_PLAYING) {
    alSourcePlay(mOpenAlId);
  }

  return true;
}

} // namespace cs::audio