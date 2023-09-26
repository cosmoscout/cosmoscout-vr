////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "AudioEngine.hpp"

#include "../cs-audio/OpenAlManager.hpp"

namespace cs::core {

////////////////////////////////////////////////////////////////////////////////////////////////////

AudioEngine::AudioEngine(std::shared_ptr<Settings> settings) 
    : mSettings(std::move(settings)) 
    , mOpenAlManager(std::make_unique<audio::OpenAlManager>(mSettings->mAudio)) {

  // Tell the user what's going on.
  logger().debug("Creating AudioEngine.");
  // logger().info("OpenAL-Soft Vendor:  {}", glGetString(GL_VENDOR));
  // logger().info("OpenAL-Soft Version: {}", glGetString(GL_VERSION));
  
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cs::core
