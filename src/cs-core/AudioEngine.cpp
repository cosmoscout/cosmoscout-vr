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
#include "../cs-audio/test_utils.hpp"
#include "../cs-audio/internal/BufferManager.hpp"
#include "../cs-audio/internal/ProcessingStepsManager.hpp"
#include "../cs-audio/internal/alErrorHandling.hpp"
#include "../cs-audio/internal/UpdateConstructor.hpp"
#include "../cs-utils/Property.hpp"

// for testing:
#include <any>
#include <cmath>
#include <map>

namespace cs::core {

std::shared_ptr<AudioEngine> AudioEngine::createAudioEngine(std::shared_ptr<Settings> settings, 
  std::shared_ptr<SolarSystem> solarSystem, std::shared_ptr<GuiManager> guiManager) {

  static auto audioEngine = std::shared_ptr<AudioEngine>(new AudioEngine(settings, solarSystem, guiManager));
  return audioEngine;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

AudioEngine::AudioEngine(std::shared_ptr<Settings> settings, std::shared_ptr<SolarSystem> solarSystem,
  std::shared_ptr<GuiManager> guiManager) 
    : mSettings(std::move(settings)) 
    , mGuiManager(std::move(guiManager))
    , mOpenAlManager(audio::OpenAlManager::createOpenAlManager())
    , mBufferManager(audio::BufferManager::createBufferManager()) 
    , mProcessingStepsManager(audio::ProcessingStepsManager::createProcessingStepsManager(mSettings))
    , mObserver(solarSystem->getObserver())
    , mSolarSystem(std::move(solarSystem))
    , mMasterVolume(utils::Property<float>(1.f))
    , mUpdateConstructor(audio::UpdateConstructor::createUpdateConstructor(mProcessingStepsManager)) {

  // Tell the user what's going on.
  logger().debug("Creating AudioEngine.");

  if (!mOpenAlManager->initOpenAl(mSettings->mAudio)) {
    logger().error("Failed to (fully) initialize OpenAL!");
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

std::vector<std::string> AudioEngine::getDevices() {
  return mOpenAlManager->getDevices();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool AudioEngine::setDevice(std::string outputDevice) {
  if (mOpenAlManager->setDevice(outputDevice)) {
    // update gui:
    mGuiManager->getGui()->callJavascript("CosmoScout.gui.setDropdownValue", 
      "audio.outputDevice", outputDevice); 
    return true;
  }
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
  mGuiManager->addSettingsSectionToSideBarFromHTML("Audio", "volume_up",
      "../share/resources/gui/audio_settings.html");
  mGuiManager->executeJavascriptFile("../share/resources/gui/js/audio_settings.js"); 

  // register callback for master volume slider
  mGuiManager->getGui()->registerCallback("audio.masterVolume",
      "Values sets the overall audio volume.", std::function([this](double value) {
        setMasterVolume(static_cast<float>(value));
      }));
  mMasterVolume.connectAndTouch(
      [this](float value) { mGuiManager->setSliderValue("audio.masterVolume", value); }); 

  // Fill the dropdowns with the available output devices
  for (auto device : getDevices()) {
    mGuiManager->getGui()->callJavascript("CosmoScout.gui.addDropdownValue",
        "audio.outputDevice", device, device.substr(14, device.length()), false);
  }

  // register callback for dropdown output devices
  mGuiManager->getGui()->registerCallback("audio.outputDevice",
      "Sets the audio output device.", std::function([this](std::string value) {
        setDevice(static_cast<std::string>(value));
      }));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void AudioEngine::update() {
  
  // Call all update functions of active Processing steps
  mProcessingStepsManager->callPsUpdateFunctions();

  // Check if a stream finished a buffer. If so refill and requeue buffer to the stream.
  for (auto controller : mAudioControllers) {
    controller->updateStreamingSources();
  }

  // Spatialization Test
  static glm::dvec3 coordinates1(-588086.8558471624, 3727313.5198930562, 10001091.473068066);
  static glm::dvec3 coordinates2(1326020.4340910933, 4085028.8301154007, 4703445.134519765);
  auto celesObj = mSolarSystem->getObject("Earth");
  if (celesObj == nullptr) { return; }
  
  static int y = 0;
  if (y % 60 == 0) {
    std::cout << "defined source length: " << glm::length(coordinates1) << std::endl;
  }
  y++;
  glm::dvec3 sourceRelPosToObs1 = celesObj->getObserverRelativePosition(coordinates1);
  sourceRelPosToObs1 *= static_cast<float>(mSolarSystem->getObserver().getScale());
  testSourcePosition1->set("position", sourceRelPosToObs1);
  testSourcePosition1->set("observerScale", mSolarSystem->getObserver().getScale());

  // glm::dvec3 sourceRelPosToObs2 = celesObj->getObserverRelativePosition(coordinates2);
  // sourceRelPosToObs2 *= static_cast<float>(mSolarSystem->getObserver().getScale());
  // testSourcePosition2->set("position", sourceRelPosToObs2);

  controllerSpace->update();

  /*
  // Streaming Test
  static bool x = true;
  if (x) {
    logger().debug("play streaming");
    testSourceStreaming->play();
    controllerAmbient->update();
    x = false;

    ALint state;
    alGetSourcei(testSourceStreaming->getOpenAlId(), AL_SOURCE_STATE, &state);
    switch(state) {
      case AL_PLAYING:
        logger().debug("playing");
      case AL_PAUSED:
        logger().debug("pause");

      case AL_STOPPED:
        logger().debug("stop");

      case AL_INITIAL:
        logger().debug("init");
    }
  }
  */
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::shared_ptr<audio::AudioController> AudioEngine::createAudioController() {
  auto controller = std::make_shared<audio::AudioController>(mBufferManager, 
    mProcessingStepsManager, mUpdateConstructor);
  controller->setPipeline(std::vector<std::string>());
  mAudioControllers.push_back(controller);
  return controller;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void AudioEngine::playAmbient() {

  // Spatialization Test
  controllerAmbient = createAudioController();
  controllerAmbient->set("looping", true);
  controllerAmbient->setPipeline(std::vector<std::string>{"DirectPlay"});

  controllerSpace = createAudioController();
  controllerSpace->set("looping", true);
  controllerSpace->setPipeline(std::vector<std::string>{"ScaledSphereSpatialization", "DirectPlay"});

  testSourcePosition1 = controllerSpace->createSource("C:/Users/sass_fl/audioCS/audioCSNotes/testFiles/exotic_mono.wav");
  testSourcePosition1->play();

  // testSourcePosition2 = controllerSpace->createSource("C:/Users/sass_fl/audioCS/audioCSNotes/testFiles/alarm_mono.wav");
  // testSourcePosition2->play();

  testSourceAmbient = controllerAmbient->createSource("C:/Users/sass_fl/audioCS/audioCSNotes/testFiles/scifi_stereo.wav"); 
  // testSourceAmbient->play();
  
  controllerAmbient->update();
  controllerSpace->update();

  // Streaming Test
  /*
  controllerAmbient = createAudioController();
  controllerAmbient->set("looping", true);
  controllerAmbient->setPipeline(std::vector<std::string>{"DirectPlay"});

  testSourceStreaming = controllerAmbient->createStreamingSource("C:/Users/sass_fl/audioCS/audioCSNotes/testFiles/scifi_stereo.wav");
  */
}

} // namespace cs::core