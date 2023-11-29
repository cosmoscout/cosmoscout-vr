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

namespace cs::core {

AudioEngine::AudioEngine(std::shared_ptr<Settings> settings, std::shared_ptr<GuiManager> guiManager) 
  : std::enable_shared_from_this<AudioEngine>()
  , mSettings(std::move(settings)) 
  , mGuiManager(std::move(guiManager))
  , mOpenAlManager(std::make_shared<audio::OpenAlManager>())// audio::OpenAlManager::createOpenAlManager())
  , mBufferManager(std::make_shared<audio::BufferManager>())// audio::BufferManager::createBufferManager()) 
  , mProcessingStepsManager(std::make_shared<audio::ProcessingStepsManager>(mSettings))// audio::ProcessingStepsManager::createProcessingStepsManager(mSettings))
  , mUpdateConstructor(std::make_shared<audio::UpdateConstructor>(mProcessingStepsManager))// audio::UpdateConstructor::createUpdateConstructor(mProcessingStepsManager))
  , mMasterVolume(utils::Property<float>(1.f)) 
  , mAudioControllers(std::vector<std::weak_ptr<audio::AudioController>>()) {

  // Tell the user what's going on.
  logger().debug("Creating AudioEngine.");

  if (!mOpenAlManager->initOpenAl(mSettings->mAudio)) {
    logger().error("Failed to (fully) initialize OpenAL!");
    return;
  }
  logger().info("OpenAL-Soft Vendor:  {}", alGetString(AL_VENDOR));
  logger().info("OpenAL-Soft Version:  {}", alGetString(AL_VERSION));

  createGUI();
}
 
AudioEngine::~AudioEngine() {
  std::cout << "close AudioEngine" << std::endl;
  try {
    // Tell the user what's going on.
    logger().debug("Deleting AudioEngine.");

    mAudioControllers.clear();
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
  bool controllerExpired = false;

  for (auto controller : mAudioControllers) {
    if (controller.expired()) {
      controllerExpired = true;
      continue;
    }
    controller.lock()->updateStreamingSources();
  }
  if (controllerExpired) {
    mAudioControllers.erase(std::remove_if(mAudioControllers.begin(), mAudioControllers.end(),
      [](const std::weak_ptr<audio::AudioController>& ptr) {
          return ptr.expired();
      }),
      mAudioControllers.end());
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::shared_ptr<audio::AudioController> AudioEngine::createAudioController() {
  static int controllerId = 0;
  auto controller = std::make_shared<audio::AudioController>(mBufferManager, 
    mProcessingStepsManager, mUpdateConstructor, controllerId++);
  controller->setPipeline(std::vector<std::string>{"DirectPlay"});
  mAudioControllers.push_back(controller);
  return controller;
}

} // namespace cs::core