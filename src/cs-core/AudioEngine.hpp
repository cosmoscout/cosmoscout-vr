////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CS_CORE_AUDIO_AUDIOENGINE_HPP
#define CS_CORE_AUDIO_AUDIOENGINE_HPP

#include "cs_audio_export.hpp"
#include "Settings.hpp"
#include "SolarSystem.hpp"
#include "GuiManager.hpp"

#include "../cs-audio/internal/OpenAlManager.hpp"
#include "../cs-audio/Source.hpp"
#include "../cs-audio/SourceGroup.hpp"
#include "../cs-audio/AudioController.hpp"
#include "../cs-audio/internal/BufferManager.hpp"
#include "../cs-audio/internal/ProcessingStepsManager.hpp"
#include "../cs-utils/Property.hpp"

#include <map>
#include <any>

namespace cs::core {

class CS_CORE_EXPORT AudioEngine {

 public:
  AudioEngine(const AudioEngine& obj) = delete;
  AudioEngine(AudioEngine&&) = delete;

  AudioEngine& operator=(const AudioEngine&) = delete;
  AudioEngine& operator=(AudioEngine&&) = delete;

  static std::shared_ptr<AudioEngine> createAudioEngine(std::shared_ptr<Settings> settings, 
  std::shared_ptr<SolarSystem> solarSystem, std::shared_ptr<GuiManager> guiManager);
  ~AudioEngine();

  /// Returns a list of all possible Output Devices 
  std::vector<std::string> getDevices();
  /// Sets the output device for the audioEngine
  bool setDevice(std::string outputDevice);
  /// Sets the master volume for the audioEngine 
  bool setMasterVolume(float gain);
  /// Update OpenAL Listener
  void update();
  /// Create a new AudioController
  std::shared_ptr<audio::AudioController> createAudioController();

 private:
  std::shared_ptr<core::Settings>                      mSettings;
  std::shared_ptr<audio::OpenAlManager>                mOpenAlManager;
  std::shared_ptr<audio::BufferManager>                mBufferManager;
  std::shared_ptr<audio::ProcessingStepsManager>       mProcessingStepsManager;
  cs::scene::CelestialObserver                         mObserver; // ?
  std::shared_ptr<core::GuiManager>                    mGuiManager; // ?
  utils::Property<float>                               mMasterVolume;
  std::vector<std::shared_ptr<audio::AudioController>> mAudioControllers;

  AudioEngine(std::shared_ptr<Settings> settings, 
    std::shared_ptr<SolarSystem> solarSystem, std::shared_ptr<GuiManager> guiManager);

  void createGUI();

  // for testing
  std::shared_ptr<audio::AudioController> audioController;
  void playAmbient();
  std::shared_ptr<audio::Source> testSourceA;
  std::shared_ptr<audio::Source> testSourceB;
  std::shared_ptr<audio::SourceGroup> testSourceGroup;
  std::map<std::string, std::any> testSettings;

  std::shared_ptr<SolarSystem> mSolarSystem; // ?
};

} // namespace cs::core

#endif // CS_CORE_AUDIO_AUDIOENGINE_HPP
