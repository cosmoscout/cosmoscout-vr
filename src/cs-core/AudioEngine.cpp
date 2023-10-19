////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "AudioEngine.hpp"
#include "Settings.hpp"
#include "SolarSystem.hpp"
#include "GuiManager.hpp"

#include "../cs-audio/internal/FileReader.hpp"
#include "../cs-audio/internal/OpenAlManager.hpp"
#include "../cs-audio/internal/Listener.hpp"
#include "../cs-audio/Source.hpp"
#include "../cs-audio/SourceGroup.hpp"
#include "../cs-audio/internal/BufferManager.hpp"
#include "../cs-audio/internal/ProcessingStepsManager.hpp"
#include "../cs-audio/internal/alErrorHandling.hpp"

// for testing:
#include <any>
#include <map>

namespace cs::core {

////////////////////////////////////////////////////////////////////////////////////////////////////

AudioEngine::AudioEngine(std::shared_ptr<Settings> settings, std::shared_ptr<SolarSystem> solarSystem,
  std::shared_ptr<GuiManager> guiManager) 
    : mSettings(std::move(settings)) 
    , mGuiManager(std::move(guiManager))
    , mOpenAlManager(std::make_unique<audio::OpenAlManager>())
    , mBufferManager(std::make_shared<audio::BufferManager>()) 
    , mProcessingStepsManager(std::make_shared<audio::ProcessingStepsManager>()) 
    , mObserver(solarSystem->getObserver())
    , mSolarSystem(std::move(solarSystem))
    , mMasterVolume(1.f) {

  // Tell the user what's going on.
  logger().debug("Creating AudioEngine.");

  if (!mOpenAlManager->initOpenAl(mSettings->mAudio)) {
    logger().warn("Failed to (fully) initialize OpenAL!");
    return;
  }
  logger().info("OpenAL-Soft Vendor:  {}", alGetString(AL_VENDOR));
  logger().info("OpenAL-Soft Version:  {}", alGetString(AL_VERSION));

  createGUI();
  playAmbient();
}
 
AudioEngine::~AudioEngine() {
  try {
    // Tell the user what's going on.
    logger().debug("Deleting AudioEngine.");
  } catch (...) {}
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

bool AudioEngine::setMasterVolume(float gain) {
  if (gain < 0) {
    logger().warn("Unable to set a negative gain!");
    return false;
  }
  alListenerf(AL_GAIN, (ALfloat) gain);
  if (audio::alErrorHandling::errorOccurred()) {
    logger().warn("Failed to set master volume!");
    return false;
  }
  mMasterVolume = gain;
  return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void AudioEngine::createGUI() {
  // add settings to GUI
  mGuiManager->addSettingsSectionToSideBarFromHTML("Audio", "accessibility_new",
      "../share/resources/gui/audio_settings.html");
  mGuiManager->executeJavascriptFile("../share/resources/gui/js/audio_settings.js"); 

  // register callback for master volume slider
  mGuiManager->getGui()->registerCallback("audio.masterVolume",
      "Values sets the overall audio volume.", std::function([this](double value) {
        setMasterVolume(static_cast<float>(value));
      }));
  mMasterVolume.connectAndTouch(
      [this](float value) { mGuiManager->setSliderValue("audio.masterVolume", value); }); 
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void AudioEngine::update() {

  static int x = 0;

  if (x % 60 == 0) {
    auto pos = mObserver.getPosition();
    std::cout << "observer pos:   " << pos.x << ", " << pos.y << ", " << pos.z << std::endl;

    std::cout << "observer speed: " << mSolarSystem->pCurrentObserverSpeed << std::endl;
  }
  ++x;

  // cs::audio::Listener::setPosition();
  // cs::audio::Listener::setVelocity();
  // cs::audio::Listener::setOrientation();
}

/////////////////////////////////////////////////////////////////////// /////////////////////////////

void AudioEngine::createAudioController() {
  auto controller = std::make_shared<audio::AudioController>(mBufferManager, mProcessingStepsManager);
  mAudioControllers.push_back(controller);
  return controller;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void AudioEngine::playAmbient() {
  audioController = createAudioController();

  testSourceA = audioController->createSource("C:/Users/sass_fl/audioCS/audioCSNotes/testFiles/scifi_stereo.wav"); 
  testSourceB = audioController->createSource("C:/Users/sass_fl/audioCS/audioCSNotes/testFiles/exotic_mono.wav");

  testSourceA->play();
  testSourceB->play();

  testSourceGroup = audioController->createSourceGroup();
  testSourceGroup->add(testSourceA);
  testSourceGroup->add(testSourceB);
  
  testSourceGroup->set("looping", true);
  audioController->set("pitch", 3.0f);
  testSourceA->set("pitch", 1.0f);

  audioController->update(); 
}

} // namespace cs::core