////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CS_AUDIO_UPDATE_INSTRUCTOR_HPP
#define CS_AUDIO_UPDATE_INSTRUCTOR_HPP

#include "cs_audio_export.hpp"
#include <iostream>
#include <memory>
#include <set>
#include <vector>

namespace cs::audio {

// forward declarations
class SourceBase;
class SourceGroup;
class AudioController;

/**
 * @brief This class acts as the manager telling who needs to be updated. Each audio controller
 * has its own updateInstructor. When a SourceSettings instance gets updated, it will register
 * itself to this class. When AudioController::update() gets called, the UpdateInstructor will
 * creates the update instructions, containing all sourceSettings instances, which need to be
 *updated, and their update scope. There are 3 update scopes:
 *  1. updateAll: When updating the audioController settings. The audioController, all Groups and
 *Source get processed
 *  2. updateWithGroup: When updating a Group. All changed groups and all their members get
 *processed
 *  3. updateSourceOnly: When updating a Source. Only the changed source gets processed
 * When updateAll is active updateWithGroup and updateSourceOnly get ignored because all sources and
 *groups need to be processed anyways. Otherwise both will be used to determine the sources which
 *need to be updated. There is a filtering step to ensure that no source is part of both update
 *scopes.
 **/
class CS_AUDIO_EXPORT UpdateInstructor {
 public:
  UpdateInstructor();
  ~UpdateInstructor();

  /// @brief Adds a Source to the updateList
  /// @param source Source to add
  void update(std::shared_ptr<SourceBase> source);

  /// @brief Adds a Source Group, and therefor all Member Sources, to the updateList
  /// @param sourceGroup Source Group to add
  void update(std::shared_ptr<SourceGroup> sourceGroup);

  /// @brief Adds an AudioController, and therefor all Sources and Groups
  /// which live on the controller, to the updateList.
  /// @param audioController AudioController to add
  void update(std::shared_ptr<AudioController> audioController);

  /// @brief Removes a Source from the updateList
  /// @param source Source to remove
  void removeUpdate(std::shared_ptr<SourceBase> source);

  /// @brief Removes a Source Group, and therefor all Member Sources, from the updateList
  /// @param sourceGroup Source Group to remove
  void removeUpdate(std::shared_ptr<SourceGroup> sourceGroup);

  /// @brief Removes an AudioController, and therefor all Sources and Groups
  /// which live on the controller, from the updateList.
  /// @param audioController AudioController to remove
  void removeUpdate(std::shared_ptr<AudioController> audioController);

  /// Struct to hold all update instructions
  struct UpdateInstruction {
    bool                                                      updateAll;
    std::shared_ptr<std::vector<std::shared_ptr<SourceBase>>> updateWithGroup;
    std::shared_ptr<std::vector<std::shared_ptr<SourceBase>>> updateSourceOnly;

    // for testing:
    void print() {
      std::cout << "-----Update Instructions-----" << std::endl;
      std::cout << "updateAll: " << (updateAll ? "true" : "false") << std::endl;
      std::cout << "size group update: "
                << (updateWithGroup == nullptr ? 0 : updateWithGroup->size()) << std::endl;
      std::cout << "size source update: "
                << (updateSourceOnly == nullptr ? 0 : updateSourceOnly->size()) << std::endl;
      std::cout << "-----------------------------" << std::endl;
    }
  };

  /// @brief Creates the update instructions when calling AudioController::update().
  UpdateInstruction createUpdateInstruction();

 private:
  struct WeakPtrComparatorGroup {
    bool operator()(
        const std::weak_ptr<SourceGroup>& left, const std::weak_ptr<SourceGroup>& right) const {
      std::owner_less<std::shared_ptr<SourceGroup>> sharedPtrLess;
      return sharedPtrLess(left.lock(), right.lock());
    }
  };

  struct WeakPtrComparatorSource {
    bool operator()(
        const std::weak_ptr<SourceBase>& left, const std::weak_ptr<SourceBase>& right) const {
      std::owner_less<std::shared_ptr<SourceBase>> sharedPtrLess;
      return sharedPtrLess(left.lock(), right.lock());
    }
  };

  /// List of all source to be updated.
  std::set<std::weak_ptr<SourceBase>, WeakPtrComparatorSource> mSourceUpdateList;
  /// List of all source groups to be updated.
  std::set<std::weak_ptr<SourceGroup>, WeakPtrComparatorGroup> mGroupUpdateList;
  /// Indicates if the audioController settings changed.
  bool mAudioControllerUpdate;
};

} // namespace cs::audio

#endif // CS_AUDIO_UPDATE_INSTRUCTOR_HPP
