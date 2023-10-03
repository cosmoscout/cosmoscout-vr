////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CS_CORE_AUDIO_AudioEngine_HPP
#define CS_CORE_AUDIO_AudioEngine_HPP

#include "cs_audio_export.hpp"
#include "Settings.hpp"

#include "../cs-audio/internal/OpenAlManager.hpp"
#include "../cs-audio/Source.hpp"
#include "../cs-audio/SourceSettings.hpp"

#include "../cs-audio/Pipeline.hpp"
#include "../cs-audio/processingSteps/Default_PS.hpp"
#include "../cs-audio/processingSteps/Spatialization_PS.hpp"

namespace cs::core {

class CS_CORE_EXPORT AudioEngine {

 public:
  explicit AudioEngine(std::shared_ptr<Settings> settings);

  ~AudioEngine();

  std::shared_ptr<audio::Source> createSource(std::string file, std::shared_ptr<audio::SourceSettings> settings=nullptr);
  /// Returns a list of all possible Output Devices 
  std::vector<std::string> getDevices();
  bool setDevice(std::string outputDevice);

 private:
  std::shared_ptr<core::Settings>       mSettings;
  std::unique_ptr<audio::OpenAlManager> mOpenAlManager;
  std::shared_ptr<audio::BufferManager> mBufferManager;

  // for testing
  void playAmbient(std::string file);
  std::shared_ptr<audio::Source> testSource;
  std::shared_ptr<audio::Pipeline> testPipeline;
};

} // namespace cs::core

#endif // CS_CORE_AUDIO_AudioEngine_HPP
