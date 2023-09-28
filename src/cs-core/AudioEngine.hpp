////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CS_CORE_AUDIO_AudioEngine_HPP
#define CS_CORE_AUDIO_AudioEngine_HPP

#include "Settings.hpp"

#include "../cs-audio/internal/OpenAlManager.hpp"
#include "../cs-audio/Source.hpp"

// forward declaration

namespace cs::core {

class CS_CORE_EXPORT AudioEngine {

 public:
  explicit AudioEngine(std::shared_ptr<Settings> settings);

  ~AudioEngine();

  audio::Source createSource(std::string file /*, AudioSettings*/);

 private:
  std::shared_ptr<core::Settings>       mSettings;
  std::unique_ptr<audio::OpenAlManager> mOpenAlManager;
  std::shared_ptr<audio::BufferManager> mBufferManager;
};

} // namespace cs::core

#endif // CS_CORE_AUDIO_AudioEngine_HPP
