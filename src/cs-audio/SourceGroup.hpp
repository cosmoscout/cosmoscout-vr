////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CS_CORE_AUDIO_SOURCE_GROUP_HPP
#define CS_CORE_AUDIO_SOURCE_GROUP_HPP

#include "cs_audio_export.hpp"
#include "Source.hpp"
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

  /// Add a new source to the group
  void join(std::shared_ptr<Source> source);
  /// Remove a source from the group
  void remove(std::shared_ptr<Source> source);
  /// Remove all sources form the group
  void reset();

  std::set<std::shared_ptr<Source>> getMembers() const;
    
 private:
  std::set<std::shared_ptr<Source>>  mMembers;
  std::shared_ptr<UpdateConstructor> mUpdateConstructor;
  std::shared_ptr<AudioController> mAudioController; // TODO: good idea?
  
  void addToUpdateList();
};

} // namespace cs::audio

#endif // CS_CORE_AUDIO_SOURCE_GROUP_HPP
