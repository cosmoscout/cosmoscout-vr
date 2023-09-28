////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "AudioEngine.hpp"
#include "Settings.hpp"

#include "../cs-audio/internal/OpenAlManager.hpp"
#include "../cs-audio/Source.hpp"

namespace cs::core {

////////////////////////////////////////////////////////////////////////////////////////////////////

AudioEngine::AudioEngine(std::shared_ptr<Settings> settings) 
    : mSettings(std::move(settings)) 
    , mOpenAlManager(std::make_unique<audio::OpenAlManager>(mSettings)) 
    , mBufferManager(std::make_shared<audio::BufferManager>()) {

  // Tell the user what's going on.
  logger().debug("Creating AudioEngine.");
  // logger().info("OpenAL-Soft Vendor:  {}", glGetString(GL_VENDOR));
  // logger().info("OpenAL-Soft Version: {}", glGetString(GL_VERSION));
  
}

AudioEngine::~AudioEngine() {

}

////////////////////////////////////////////////////////////////////////////////////////////////////

audio::Source AudioEngine::createSource(std::string file /*AudioSettings*/) {
  return audio::Source(mBufferManager, file);
}

} // namespace cs::core
