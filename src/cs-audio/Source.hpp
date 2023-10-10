////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CS_AUDIO_SOURCE_HPP
#define CS_AUDIO_SOURCE_HPP

#include "cs_audio_export.hpp"
#include "internal/BufferManager.hpp"
#include "internal/ProcessingStepsManager.hpp"

#include <AL/al.h>
#include <map>
#include <any>

namespace cs::audio {

// forward declaration
class SourceGroup;

class CS_AUDIO_EXPORT Source {
 public:
  ~Source();
  
  bool play() const;
  bool stop() const;
  void update();
  /// Sets settings that will be applied when calling update(). 
  void set(std::string, std::any);

  bool setFile(std::string file);
  std::string getFile() const;

  std::shared_ptr<std::map<std::string, std::any>> getSettings() const;

  // TODO: Constructor in private ausprobieren

  Source(std::shared_ptr<BufferManager> bufferManager, 
  std::shared_ptr<ProcessingStepsManager> processingStepsManager,
  std::string file);

  friend class SourceGroup;
  
  // friend class cs::core::AudioEngine;
 private:
  std::string                                      mFile;
  ALuint                                           mOpenAlId;
  std::shared_ptr<BufferManager>                   mBufferManager;
  std::shared_ptr<ProcessingStepsManager>          mProcessingStepsManager;
  std::shared_ptr<std::map<std::string, std::any>> mCurrentSettings;
  /// Contains all settings that are about to be set using the update() function. 
  /// If update() is called these settings will be used to call all the processing 
  /// steps. When finished, all set values will be written into mCurrentSettings
  /// and settings gets reset.
  std::shared_ptr<std::map<std::string, std::any>> mSettings;
};

} // namespace cs::audio

#endif // CS_AUDIO_SOURCE_HPP
