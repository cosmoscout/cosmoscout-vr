////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CS_CORE_AUDIO_AUDIOENGINE_HPP
#define CS_CORE_AUDIO_AUDIOENGINE_HPP

#include "cs_audio_export.hpp"
#include "Settings.hpp"

#include "../cs-audio/internal/OpenAlManager.hpp"
#include "../cs-audio/Source.hpp"
#include "../cs-audio/SourceGroup.hpp"
#include "../cs-audio/AudioController.hpp"
#include "../cs-audio/internal/BufferManager.hpp"
#include "../cs-audio/internal/ProcessingStepsManager.hpp"

#include <map>
#include <any>

namespace cs::core {

class CS_CORE_EXPORT AudioEngine {

 public:
  AudioEngine(const AudioEngine& obj) = delete;
  AudioEngine(AudioEngine&&) = delete;

  AudioEngine& operator=(const AudioEngine&) = delete;
  AudioEngine& operator=(AudioEngine&&) = delete;

  explicit AudioEngine(std::shared_ptr<Settings> settings);
  ~AudioEngine();

  /// Returns a list of all possible Output Devices 
  std::shared_ptr<std::vector<std::string>> getDevices();
  /// Sets the output device for the audioEngine
  bool setDevice(std::string outputDevice);
  /// Sets the master volume for the audioEngine 
  bool setMasterVolume(float gain);

 private:
  std::shared_ptr<core::Settings>                mSettings;
  std::unique_ptr<audio::OpenAlManager>          mOpenAlManager;
  std::shared_ptr<audio::BufferManager>          mBufferManager;
  std::shared_ptr<audio::ProcessingStepsManager> mProcessingStepsManager;

  void createAudioControls();

  // for testing
  std::shared_ptr<audio::AudioController> audioController;
  void playAmbient();
  std::shared_ptr<audio::Source> testSourceA;
  std::shared_ptr<audio::Source> testSourceB;
  std::shared_ptr<audio::SourceGroup> testSourceGroup;
  std::map<std::string, std::any> testSettings;
};

} // namespace cs::core

#endif // CS_CORE_AUDIO_AUDIOENGINE_HPP
