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

/// @brief This is the derived source class for non-streaming sources. This means that the whole file
/// is being read and written into the buffer. This has the benefit that buffers can be shared among
/// all non-streaming sources. This is done via the BufferManager. 
class CS_AUDIO_EXPORT Source : public SourceBase {
 public:
 /// @brief This is the standard constructor used for non-cluster mode and cluster mode leader calls 
  Source(std::shared_ptr<BufferManager> bufferManager, 
  std::string file, std::shared_ptr<UpdateInstructor> UpdateInstructor);
  /// @brief This Constructor will create a dummy source which is used when a member of a cluster
  /// tries to create a Source. Doing this will disable any functionality of this class.
  Source();
  ~Source();
  
  /// @brief Sets a new file to be played by the source.
  /// @return Whether it was successful
  bool setFile(std::string file) override;
    
 private:
  std::shared_ptr<BufferManager> mBufferManager;
};

} // namespace cs::audio

#endif // CS_AUDIO_SOURCE_HPP
