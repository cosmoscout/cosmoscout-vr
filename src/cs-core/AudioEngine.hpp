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
#include "../cs-audio/SourceSettings.hpp"
#include "../cs-audio/internal/BufferManager.hpp"
#include "../cs-audio/internal/ProcessingStepsManager.hpp"

namespace cs::core {

class CS_CORE_EXPORT AudioEngine {

 public:
  AudioEngine(const AudioEngine& obj) = delete;
  AudioEngine(AudioEngine&&) = delete;

  AudioEngine& operator=(const AudioEngine&) = delete;
  AudioEngine& operator=(AudioEngine&&) = delete;

  explicit AudioEngine(std::shared_ptr<Settings> settings);
  ~AudioEngine();

  /// Creates a new audio source
  std::shared_ptr<audio::Source> createSource(std::string file);
  /// Returns a list of all possible Output Devices 
  std::shared_ptr<std::vector<std::string>> getDevices();
  /// Sets the output device for the audioEngine
  bool setDevice(std::string outputDevice);
  /// Sets the master volume for the audioEngine 
  bool setMasterVolume(ALfloat gain);

 private:
  std::shared_ptr<core::Settings>                mSettings;
  std::unique_ptr<audio::OpenAlManager>          mOpenAlManager;
  std::shared_ptr<audio::BufferManager>          mBufferManager;
  std::shared_ptr<audio::ProcessingStepsManager> mProcessingStepsManager;

  // for testing
  void playAmbient(std::string file);
  std::shared_ptr<audio::Source> testSourceA;
  std::shared_ptr<audio::Source> testSourceB;
  std::shared_ptr<audio::SourceSettings> testSettings;
};

} // namespace cs::core

#endif // CS_CORE_AUDIO_AUDIOENGINE_HPP
