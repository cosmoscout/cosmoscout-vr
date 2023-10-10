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
#include "../cs-audio/internal/BufferManager.hpp"
#include "../cs-audio/internal/ProcessingStepsManager.hpp"
#include "../cs-audio/internal/alErrorHandling.hpp"

// for testing:
#include "../cs-audio/internal/SettingsMixer.hpp"
#include <map>
#include <any>

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

std::shared_ptr<audio::Source> AudioEngine::createSource(std::string file) {
  return std::make_shared<audio::Source>(mBufferManager, mProcessingStepsManager, file);
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
  testSourceA = createSource(file); 
  testSourceA->set("looping", true);
  testSourceA->play();

  // test SettingsMixer
  auto sourceCurrent = std::make_shared<std::map<std::string, std::any>>();
  auto sourceNew = std::make_shared<std::map<std::string, std::any>>();
  auto groupNew = std::make_shared<std::map<std::string, std::any>>();

  std::cout << "----Test 1----" << std::endl;

  sourceCurrent->operator[]("gain") = 1.5f;
  sourceCurrent->operator[]("pitch") = 1.0f;
  groupNew->operator[]("looping") = true;

  printMap(audio::SettingsMixer::mixGroupUpdate(sourceCurrent, groupNew));
  /*
  looping: true
  */
  sourceCurrent->clear();
  groupNew->clear();

  std::cout << "----Test 2----" << std::endl;
  sourceCurrent->operator[]("gain") = 1.5f;
  sourceCurrent->operator[]("pitch") = 1.0f;
  sourceCurrent->operator[]("looping") = false;
  groupNew->operator[]("looping") = true;

  printMap(audio::SettingsMixer::mixGroupUpdate(sourceCurrent, groupNew));
  /*
  -
  */
  sourceCurrent->clear();
  groupNew->clear();

  
  std::cout << "----Test 3----" << std::endl;
  sourceCurrent->operator[]("gain") = 1.5f;
  sourceCurrent->operator[]("pitch") = 1.0f;

  sourceNew->operator[]("pitch") = 0.5f;

  groupNew->operator[]("looping") = true;

  printMap(audio::SettingsMixer::mixGroupAndSourceUpdate(sourceCurrent, sourceNew, groupNew));
  /*
  pitch: 0.5
  looping: true
  */
  sourceCurrent->clear();
  sourceNew->clear();
  groupNew->clear();

  std::cout << "----Test 4----" << std::endl;
  sourceCurrent->operator[]("gain") = 1.5f;
  sourceCurrent->operator[]("pitch") = 1.0f;

  sourceNew->operator[]("pitch") = 0.5f;
  sourceNew->operator[]("looping") = false;
  groupNew->operator[]("looping") = true;

  printMap(audio::SettingsMixer::mixGroupAndSourceUpdate(sourceCurrent, sourceNew, groupNew));
  /*
  pitch: 0.5
  looping: false
  */
  sourceCurrent->clear();
  sourceNew->clear();
  groupNew->clear();
}

void AudioEngine::printMap(std::shared_ptr<std::map<std::string, std::any>> map) {
  for (auto const& [key, val] : *map) {
    std::cout << key << ": ";

    auto type = val.type().name();
    std::cout << "(" << type << ") ";
    try {
      std::cout << std::any_cast<int>(val);
    } catch (const std::bad_any_cast&)
    {
    }

    try{
      std::cout << std::any_cast<float>(val);
    } catch (const std::bad_any_cast&)
    {
    } 

    try {
      std::cout << std::any_cast<bool>(val);
    } catch (const std::bad_any_cast&)
    {
    }

    std::cout << std::endl;
  }
}

} // namespace cs::core
