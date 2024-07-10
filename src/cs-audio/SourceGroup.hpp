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

#include <any>
#include <map>
#include <memory>
#include <set>
#include <string>

namespace cs::audio {

// forward declarations
class UpdateInstructor;

/// @brief A SourceGroup is a way to apply source settings to multiple sources at once. Each
/// source can only be part of one group at a time and both *MUST* be on the same audio controller.
/// Doing otherwise can lead to undefined behavior and is not intended.
class CS_AUDIO_EXPORT SourceGroup : public SourceSettings,
                                    public std::enable_shared_from_this<SourceGroup> {

 public:
  /// @brief This is the standard constructor used for non-cluster mode and cluster mode leader
  /// calls
  explicit SourceGroup(std::shared_ptr<UpdateInstructor> UpdateInstructor,
      std::shared_ptr<UpdateConstructor>                 updateConstructor,
      std::shared_ptr<AudioController>                   audioController);
  /// @brief This Constructor will create a dummy Group which is used when a member of a cluster
  /// tries to create a Group. Doing this will disable any functionality of this class.
  explicit SourceGroup();
  ~SourceGroup();

  /// @brief Adds a new source to the group
  void join(std::shared_ptr<SourceBase> source);
  /// @brief Removes a source from the group
  void leave(std::shared_ptr<SourceBase> source);
  /// @brief Removes all sources form the group
  void reset();

  /// @return List to all members of the group
  const std::vector<std::shared_ptr<SourceBase>> getMembers();

 private:
  struct WeakPtrComparatorSource {
    bool operator()(
        const std::weak_ptr<SourceBase>& left, const std::weak_ptr<SourceBase>& right) const {
      std::owner_less<std::shared_ptr<SourceBase>> sharedPtrLess;
      return sharedPtrLess(left.lock(), right.lock());
    }
  };

  std::set<std::weak_ptr<SourceBase>, WeakPtrComparatorSource> mMembers;
  std::shared_ptr<UpdateConstructor>                           mUpdateConstructor;
  std::weak_ptr<AudioController>                               mAudioController;

  /// @brief registers itself to the updateInstructor to be updated
  void addToUpdateList() override;
  /// @brief deregister itself from the updateInstructor
  void removeFromUpdateList() override;
};

} // namespace cs::audio

#endif // CS_CORE_AUDIO_SOURCE_GROUP_HPP
