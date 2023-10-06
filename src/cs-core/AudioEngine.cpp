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
#include "../cs-audio/internal/BufferManager.hpp"
#include "../cs-audio/internal/ProcessingStepsManager.hpp"
#include "../cs-audio/internal/alErrorHandling.hpp"

namespace cs::core {

////////////////////////////////////////////////////////////////////////////////////////////////////

AudioEngine::AudioEngine(std::shared_ptr<Settings> settings) 
    : mSettings(std::move(settings)) 
    , mOpenAlManager(std::make_unique<audio::OpenAlManager>(mSettings))
    , mBufferManager(std::make_shared<audio::BufferManager>()) 
    , mProcessingStepsManager(std::make_unique<audio::ProcessingStepsManager>(mSettings)){

  // Tell the user what's going on.
  logger().debug("Creating AudioEngine.");
  logger().info("OpenAL-Soft Vendor:  {}", alGetString(AL_VENDOR));
  logger().info("OpenAL-Soft Version:  {}", alGetString(AL_VERSION));

  playAmbient("C:/Users/sass_fl/audioCS/audioCSNotes/testFiles/scifi_stereo.wav");
}
 
AudioEngine::~AudioEngine() {
  try {
    // Tell the user what's going on.
    logger().debug("Deleting AudioEngine.");
  } catch (...) {}
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::shared_ptr<audio::Source> AudioEngine::createSource(std::string file, 
 std::shared_ptr<audio::SourceSettings> settings) {
  return std::shared_ptr<audio::Source>(new audio::Source(mBufferManager, mProcessingStepsManager, file, 
    ((settings == nullptr) ? std::make_shared<audio::SourceSettings>() : settings)));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::shared_ptr<std::vector<std::string>> AudioEngine::getDevices() {
  std::shared_ptr<std::vector<std::string>> result = std::make_shared<std::vector<std::string>>();
  int macro;

  if (alcIsExtensionPresent(NULL, "ALC_ENUMERATE_ALL_EXT") == AL_TRUE) {
    macro = ALC_ALL_DEVICES_SPECIFIER;

  } else if (alcIsExtensionPresent(NULL, "ALC_ENUMERATION_EXT") == AL_TRUE) {
    logger().warn("OpenAL Extensions 'ALC_ENUMERATE_ALL_EXT' not found. Not all available devices might be found!");
    macro = ALC_DEVICE_SPECIFIER;

  } else {
    logger().warn("OpenAL Extensions 'ALC_ENUMERATE_ALL_EXT' and 'ALC_ENUMERATION_EXT' not found. Unable to find available devices!");
    return result;
  }

  const ALCchar* device = alcGetString(nullptr, macro);
  const ALCchar* next = alcGetString(nullptr, macro) + 1;
  size_t len = 0;

  while (device && *device != '\0' && next && *next != '\0') {
    result->push_back(device);
    len = strlen(device);
    device += (len + 1);
    next += (len + 2);
  }

  return result;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool AudioEngine::setDevice(std::string outputDevice) {
  return false;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool AudioEngine::setMasterVolume(ALfloat gain) {
  if (gain < 0) {
    return false;
  }
  alListenerf(AL_GAIN, gain);
  if (audio::alErrorHandling::errorOccurd()) {
    logger().warn("Failed to set master volume!");
    return false;
  }
  return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void AudioEngine::playAmbient(std::string file) {
  // testSettings = std::make_shared<audio::SourceSettings>();
  // testSettings->looping = true;
  // testSettings->gain = 1.0f;
  // testSource->mSettings->pitch = 4.0f;
  // testSource->update();

  testSettings = std::make_shared<audio::SourceSettings>();
  testSettings->looping = true;
  
  testSourceA = createSource(file, testSettings); 
  testSourceB = createSource("C:/Users/sass_fl/audioCS/audioCSNotes/testFiles/exotic_mono.wav", testSettings);

  testSourceA->play();
  testSourceB->play();

  testSettings->pitch = 4.0f; 

  testSourceA->update();
  testSourceB->update();
}

} // namespace cs::core
