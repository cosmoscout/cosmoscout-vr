////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CS_AUDIO_SOURCE_HPP
#define CS_AUDIO_SOURCE_HPP

#include "cs_audio_export.hpp"
#include "internal/SourceSettings.hpp"
#include "internal/BufferManager.hpp"
#include "internal/ProcessingStepsManager.hpp"

#include <AL/al.h>
#include <map>
#include <any>

namespace cs::audio {

// forward declaration
class SourceGroup;

class CS_AUDIO_EXPORT Source : public SourceSettings{
 public:
  ~Source();
  
  bool play() const;
  bool stop() const;

  bool setFile(std::string file);
  std::string getFile() const;

  // TODO: Constructor in private ausprobieren

  Source(std::shared_ptr<BufferManager> bufferManager, 
  std::shared_ptr<ProcessingStepsManager> processingStepsManager,
  std::string file);

  // friend class cs::core::AudioEngine;
  friend class SourceGroup;
  friend class AudioController;
  
 private:
  std::string                                      mFile;
  ALuint                                           mOpenAlId;
  std::shared_ptr<BufferManager>                   mBufferManager;
  std::shared_ptr<ProcessingStepsManager>          mProcessingStepsManager;
  std::shared_ptr<SourceGroup>                     mGroup;
};

} // namespace cs::audio

#endif // CS_AUDIO_SOURCE_HPP
