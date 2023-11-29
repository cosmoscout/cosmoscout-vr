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
  /// @return Pointer to the new source
  std::shared_ptr<StreamingSource> createStreamingSource(std::string file, 
    int bufferSize=8192, int queueSize=4);

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

  /// @return Return a list of all sources which live on the audioController
  std::vector<std::shared_ptr<SourceBase>> getSources();

  std::vector<std::shared_ptr<SourceGroup>> getGroups();

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

  template<typename T> 
  void removeExpiredElements(std::vector<std::weak_ptr<T>> elements);
};

} // namespace cs::audio

#endif // CS_AUDIO_AUDIO_CONTROLLER_HPP
