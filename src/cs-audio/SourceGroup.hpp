////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CS_CORE_AUDIO_SOURCE_GROUP_HPP
#define CS_CORE_AUDIO_SOURCE_GROUP_HPP

#include "cs_audio_export.hpp"
#include "Source.hpp"

#include <memory>
#include <string>
#include <any>
#include <map>
#include <set>

namespace cs::audio {

class CS_AUDIO_EXPORT SourceGroup : public SourceSettings
{
 public:
  SourceGroup(std::shared_ptr<ProcessingStepsManager> processingStepsManager);

  /// Add a new source to the group
  void add(std::shared_ptr<Source> source);
  /// Remove a source from the group
  void remove(std::shared_ptr<Source> source);
  /// Remove all sources form the group
  void reset();
  /// Update the group settings
  void update();
  /// Update the group settings and all member sources
  void updateAll();
  /// Update only the member sources
  void updateMembersOnly();

  /// Set settings that will be applied when calling update(). 
  void set(std::string key, std::any value);

 private:
  std::shared_ptr<std::map<std::string, std::any>> mSettings;
  std::shared_ptr<std::map<std::string, std::any>> mCurrentSettings;
  std::set<std::shared_ptr<Source>>                mMemberSources;
  std::shared_ptr<ProcessingStepsManager>          mProcessingStepsManager;
};

} // namespace cs::audio

#endif // CS_CORE_AUDIO_SOURCE_GROUP_HPP
