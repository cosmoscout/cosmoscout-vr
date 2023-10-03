////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "AudioEngine.hpp"
#include "Settings.hpp"

#include "../cs-audio/internal/FileReader.hpp"
#include "../cs-audio/internal/OpenAlManager.hpp"
#include "../cs-audio/internal/Listener.hpp"
#include "../cs-audio/Source.hpp"
#include "../cs-audio/SourceSettings.hpp"

#include "../cs-audio/Pipeline.hpp"
#include "../cs-audio/processingSteps/Default_PS.hpp"
#include "../cs-audio/processingSteps/Spatialization_PS.hpp"

namespace cs::core {

////////////////////////////////////////////////////////////////////////////////////////////////////

AudioEngine::AudioEngine(std::shared_ptr<Settings> settings) 
    : mSettings(std::move(settings)) 
    , mOpenAlManager(std::make_unique<audio::OpenAlManager>(mSettings))
    , mBufferManager(std::make_shared<audio::BufferManager>()) {

  // Tell the user what's going on.
  logger().debug("Creating AudioEngine.");
  logger().info("OpenAL-Soft Vendor:  {}", alGetString(AL_VENDOR));
  logger().info("OpenAL-Soft Version:  {}", alGetString(AL_VERSION));

  playAmbient("I:/Bachelorarbeit/audioCS/audioCSNotes/testFiles/scifi_stereo.wav");
}
 
AudioEngine::~AudioEngine() {
  try {
    // Tell the user what's going on.
    logger().debug("Deleting AudioEngine.");
  } catch (...) {}
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::shared_ptr<audio::Source> AudioEngine::createSource(std::string file, std::shared_ptr<audio::SourceSettings> settings) {
  return std::make_shared<audio::Source>(mBufferManager, file, settings);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::vector<std::string> AudioEngine::getDevices() {
  if (alcIsExtensionPresent(NULL, "ALC_ENUMERATE_ALL_EXT") == AL_TRUE) {
    logger().info("Available Devices: {}.", alcGetString(nullptr, ALC_ALL_DEVICES_SPECIFIER)); 

  } else if (alcIsExtensionPresent(NULL, "ALC_ENUMERATION_EXT") == AL_TRUE) {
    logger().warn("OpenAL Extensions 'ALC_ENUMERATE_ALL_EXT' not found. Not all available devices might be found!");
    logger().info("Available Devices: {}.", alcGetString(nullptr, ALC_DEVICE_SPECIFIER));

  } else {
    logger().warn("OpenAL Extensions 'ALC_ENUMERATE_ALL_EXT' and 'ALC_ENUMERATION_EXT' not found. Unable to find available devices!");
  }
  return std::vector<std::string>();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool AudioEngine::setDevice(std::string outputDevice) {
  return false;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void AudioEngine::playAmbient(std::string file) {
  testSource = createSource(file);
  /*
  std::shared_ptr<audio::Default_PS> default_ps = std::make_shared<audio::Default_PS>();
  std::shared_ptr<audio::Spatialization_PS> spat_ps = std::make_shared<audio::Spatialization_PS>();
  
  testPipeline = std::make_shared<audio::Pipeline>();
  */
  testSource->play();
}

} // namespace cs::core
