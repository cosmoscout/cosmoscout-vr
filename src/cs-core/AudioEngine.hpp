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


namespace cs::core {

class CS_CORE_EXPORT AudioEngine {

 public:
  explicit AudioEngine(std::shared_ptr<Settings> settings);

  ~AudioEngine();

  std::shared_ptr<audio::Source> createSource(std::string file, std::shared_ptr<audio::SourceSettings> settings=nullptr);

 private:
  std::shared_ptr<core::Settings>       mSettings;
  std::unique_ptr<audio::OpenAlManager> mOpenAlManager;
  std::shared_ptr<audio::BufferManager> mBufferManager;

  // for testing
  void playAmbient(std::string file);
  void playAmbient2();
  ALuint sources[1];
  ALuint buffer[1];
};

} // namespace cs::core

#endif // CS_CORE_AUDIO_AudioEngine_HPP
