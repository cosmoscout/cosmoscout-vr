////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CS_AUDIO_SOURCE_HPP
#define CS_AUDIO_SOURCE_HPP

#include "cs_audio_export.hpp"

#include "internal/BufferManager.hpp"
#include <AL/al.h>

// forward declaration
class AudioEngine;

namespace cs::audio {

class CS_AUDIO_EXPORT Source {
 public:
  ~Source();
  
  bool play();
  bool stop();
  void update(/*AudioSettings*/);

  bool setFile(std::string file);
  std::string getFile() const;

  friend class AudioEngine;
  Source(std::shared_ptr<BufferManager> bufferManager, std::string file /*, AudioSettings*/);
 private:
  

  std::string                    mFile;
  ALuint                         mOpenAlId;
  std::shared_ptr<BufferManager> mBufferManager;
};

} // namespace cs::audio

#endif // CS_AUDIO_SOURCE_HPP
