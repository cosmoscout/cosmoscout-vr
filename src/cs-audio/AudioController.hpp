////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CS_AUDIO_AUDIO_CONTROLLER_HPP
#define CS_AUDIO_AUDIO_CONTROLLER_HPP

#include "cs_audio_export.hpp"
#include "Source.hpp"
#include "StreamingSource.hpp"
#include "SourceGroup.hpp"
#include "internal/SourceBase.hpp"
#include "internal/BufferManager.hpp"
#include "internal/UpdateInstructor.hpp"

#include <memory>
#include <map>
#include <any>
#include <string>

namespace cs::audio {

// forward declarations
class ProcessingStepsManager;

/// @brief This class is the gateway to create audio objects and to optionally define a processing
/// pipeline for these objects. It is recommended that each use case for audio should have
/// it's own AudioController, for example each plugin should have it's own and/or a separation of
/// different sources, like spatialized sources in space and ambient background music. This is recommended
/// because each use case will most probably require a different pipeline, which if configured correctly, could
/// benefit performance.  
class CS_AUDIO_EXPORT AudioController 
  : public SourceSettings
  , public std::enable_shared_from_this<AudioController> {
    
 public:
  AudioController(
    std::shared_ptr<BufferManager> bufferManager, 
    std::shared_ptr<ProcessingStepsManager> processingStepsManager,
    std::shared_ptr<UpdateConstructor> updateConstructor,
    int id);
  ~AudioController();

  /// @brief Creates a new audio source
  /// @return Pointer to the new source
  std::shared_ptr<Source> createSource(std::string file);

  /// @brief Creates a new streaming audio source
  /// @param file audio file to stream
  /// @param bufferLength time in milliseconds of each buffer
  /// @param queueSize number of buffers used for the stream
  /// @return Pointer to the new source
  std::shared_ptr<StreamingSource> createStreamingSource(std::string file, 
    int bufferLength=200, int queueSize=4);

  /// @brief Creates a new audio source group
  /// @return Pointer to the new source group
  std::shared_ptr<SourceGroup> createSourceGroup();

  /// @brief Defines a new pipeline for the audioController
  /// @param processingSteps list of all processing steps that should be part of the pipeline
  void setPipeline(std::vector<std::string> processingSteps);

  /// @brief Calls the pipeline for all newly set settings for the audioController, Groups and Sources since
  /// the last update call.
  void update();

  void updateStreamingSources();

  /// @return A list of all sources which live on the audioController
  std::vector<std::shared_ptr<SourceBase>> getSources();

  /// @return A list of all groups which live on the audioController
  std::vector<std::shared_ptr<SourceGroup>> getGroups();

  /// @return ID of the controller. Only useful for internal AudioEngine stuff. 
  const int getControllerId() const;

 private:
  const int                                     mControllerId;
  /// Ptr to the single BufferManager of the audioEngine
  std::shared_ptr<BufferManager>                mBufferManager;
  /// Ptr to the single ProcessingStepsManager of the audioEngine
  std::shared_ptr<ProcessingStepsManager>       mProcessingStepsManager;
  /// List of all Sources that live on the AudioController
  std::vector<std::weak_ptr<SourceBase>>        mSources;
  /// List of Streaming Sources that live on the AudioController
  std::vector<std::weak_ptr<StreamingSource>>   mStreams;
  /// List of all Groups that live on the AudioController
  std::vector<std::weak_ptr<SourceGroup>>       mGroups;
  /// Ptr to the UpdateInstructor. Each AudioController has their own Instructor
  std::shared_ptr<UpdateInstructor>             mUpdateInstructor;
  /// Ptr to the single UpdateConstructor of the audioEngine
  std::shared_ptr<UpdateConstructor>            mUpdateConstructor;

  /// @brief registers itself to the updateInstructor to be updated 
  void addToUpdateList() override;
  /// @brief deregister itself from the updateInstructor 
  void removeFromUpdateList() override;

  /// @brief Removes expired weak_ptr from a vector.
  /// @tparam T SourceBase, StreamingSource, SourceGroup
  /// @param elements vector to remove from
  template<typename T> 
  void removeExpiredElements(std::vector<std::weak_ptr<T>> elements);
};

} // namespace cs::audio

#endif // CS_AUDIO_AUDIO_CONTROLLER_HPP
