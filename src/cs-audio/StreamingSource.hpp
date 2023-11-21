////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CS_AUDIO_STREAMING_SOURCE_HPP
#define CS_AUDIO_STREAMING_SOURCE_HPP

#include "cs_audio_export.hpp"
#include "internal/SourceBase.hpp"
#include "internal/BufferManager.hpp"
#include "internal/UpdateInstructor.hpp"
#include "internal/FileReader.hpp"

#include <AL/al.h>
#include <map>
#include <any>

namespace cs::audio {

// forward declaration
class SourceGroup;

class CS_AUDIO_EXPORT StreamingSource : public SourceBase {
    
 public:
  ~StreamingSource();

  /// @brief Sets a new file to be played by the source.
  /// @return true if successful
  bool setFile(std::string file) override;

  void updateStream();  

  StreamingSource(std::string file, int bufferSize, int queueSize,
    std::shared_ptr<UpdateInstructor> UpdateInstructor);

  friend class SourceGroup;
  friend class UpdateConstructor;
    
 private:
  std::vector<ALuint>   mBuffers;
  WavContainerStreaming mWavContainer;
  int mBufferSize;
};

} // namespace cs::audio

#endif // CS_AUDIO_STREAMING_SOURCE_HPP