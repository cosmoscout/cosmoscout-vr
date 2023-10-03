////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "Source.hpp"
#include "../cs-core/AudioEngine.hpp"
#include "internal/BufferManager.hpp"

#include <AL/al.h>

namespace cs::audio {

Source::Source(std::shared_ptr<BufferManager> bufferManager, std::string file, std::shared_ptr<SourceSettings> settings) 
  : mFile(std::move(file)) 
  , mBufferManager(std::move(bufferManager)) 
  , mSettings(std::move(settings)) {

  // generate new source  
  alGenSources((ALuint)1, &mOpenAlId);

  // TODO: check if file actually exists
 
  // temp for ambient
  alSource3i(mOpenAlId, AL_POSITION, 0, 0, 0);
  alSourcei(mOpenAlId, AL_LOOPING, AL_TRUE);

  // get buffer and bind buffer to source
  alSourcei(mOpenAlId, AL_BUFFER, mBufferManager->getBuffer(mFile));
  // TODO: Error handling
}

////////////////////////////////////////////////////////////////////////////////////////////////////

Source::~Source() {
  mBufferManager->removeBuffer(mFile);
  alDeleteSources(1, &mOpenAlId);
  // TODO: Error handling
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool Source::play() {
  int playing;
  alGetSourcei(mOpenAlId, AL_SOURCE_STATE, &playing);
  std::cout << "is playing: " << (playing == AL_PLAYING ? "yes" : "no") << std::endl;

  alSourcePlay(mOpenAlId);
  // TODO: Error handling

  float x, y, z;
  alGetSource3f(mOpenAlId, AL_POSITION, &x, &y, &z);
  std::cout << "source Position: " << x << ", " << y << ", " << z << std::endl;

  float gain;
  alGetSourcef(mOpenAlId, AL_GAIN, &gain);
  std::cout << "gain: " << gain << std::endl;

  int buffer;
  alGetSourcei(mOpenAlId, AL_BUFFER, &buffer);
  std::cout << "buffer: " << buffer << std::endl;

  alGetSourcei(mOpenAlId, AL_SOURCE_STATE, &playing);
  std::cout << "is playing: " << (playing == AL_PLAYING ? "yes" : "no") << std::endl;

  return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool Source::stop() {
  alSourceStop(mOpenAlId);
  // TODO: Error handling
  return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Source::update() {
    
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool Source::setFile(std::string file) {
  mBufferManager->removeBuffer(mFile);
  mFile = file;
  // TODO: check if file exists
  return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::string Source::getFile() const {
  return mFile;   
}

} // namespace cs::audio