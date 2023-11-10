////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CS_CORE_AUDIO_SOURCE_GROUP_HPP
#define CS_CORE_AUDIO_SOURCE_GROUP_HPP

#include "cs_audio_export.hpp"
#include "internal/SourceBase.hpp"
#include "internal/SourceSettings.hpp"

#include <memory>
#include <string>
#include <any>
#include <map>
#include <set>

namespace cs::audio {

// forward declarations
class UpdateInstructor;

class CS_AUDIO_EXPORT SourceGroup 
  : public SourceSettings
  , public std::enable_shared_from_this<SourceGroup> {
    
 public:
  explicit SourceGroup(std::shared_ptr<UpdateInstructor> UpdateInstructor,
    std::shared_ptr<UpdateConstructor> updateConstructor,
    std::shared_ptr<AudioController> audioController);
  ~SourceGroup();

  /// @brief Adds a new source to the group
  void join(std::shared_ptr<SourceBase> source);
  /// @brief Removes a source from the group
  void remove(std::shared_ptr<SourceBase> source);
  /// @brief Removes all sources form the group
  void reset();

  /// @return List to all members of the group
  std::set<std::shared_ptr<SourceBase>> getMembers() const;
    
 private:
  std::set<std::shared_ptr<SourceBase>>  mMembers;
  std::shared_ptr<UpdateConstructor> mUpdateConstructor;
  std::shared_ptr<AudioController> mAudioController; // TODO: good idea?
  
  /// @brief registers itself to the updateInstructor to be updated 
  void addToUpdateList();
};

} // namespace cs::audio

#endif // CS_CORE_AUDIO_SOURCE_GROUP_HPP
