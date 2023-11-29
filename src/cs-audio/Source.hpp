////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CS_AUDIO_SOURCE_HPP
#define CS_AUDIO_SOURCE_HPP

#include "cs_audio_export.hpp"
#include "internal/SourceBase.hpp"
#include "internal/BufferManager.hpp"

#include <AL/al.h>
#include <map>
#include <any>

namespace cs::audio {

// forward declaration
class SourceGroup;

class CS_AUDIO_EXPORT Source : public SourceBase {

 public:
  Source(std::shared_ptr<BufferManager> bufferManager, 
  std::string file, std::shared_ptr<UpdateInstructor> UpdateInstructor);
  ~Source();
  
  /// @brief Sets a new file to be played by the source.
  /// @return Whether it was successful
  bool setFile(std::string file) override;
    
 private:
  std::shared_ptr<BufferManager>                   mBufferManager;
};

} // namespace cs::audio

#endif // CS_AUDIO_SOURCE_HPP
