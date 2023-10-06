////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CS_AUDIO_SOURCE_HPP
#define CS_AUDIO_SOURCE_HPP

#include "cs_audio_export.hpp"
#include "internal/BufferManager.hpp"
#include "SourceSettings.hpp"
#include "internal/ProcessingStepsManager.hpp"

#include <AL/al.h>

// forward declaration
// class cs::core::AudioEngine;

namespace cs::audio {

class CS_AUDIO_EXPORT Source {
 public:
  ~Source();
  
  bool play() const;
  bool stop() const;
  void update();

  bool setFile(std::string file);
  std::string getFile() const;

  std::shared_ptr<SourceSettings> getSettings() const;

  // TODO: Constructor in private ausprobieren

  /// Contains all settings that are about to be set using the update() function. 
  /// If update() is called these settings will be used to call all the processing 
  /// steps. When finished, all set values will be written into mCurrentSettings
  /// and settings gets reset.
  std::shared_ptr<SourceSettings>         settings;
  
  Source(std::shared_ptr<BufferManager> bufferManager, 
  std::shared_ptr<ProcessingStepsManager> processingStepsManager,
  std::string file, std::shared_ptr<SourceSettings> settings);
  
  // friend class cs::core::AudioEngine;
 private:
  std::shared_ptr<SourceSettings>         mCurrentSettings;
};

} // namespace cs::audio

#endif // CS_AUDIO_SOURCE_HPP
