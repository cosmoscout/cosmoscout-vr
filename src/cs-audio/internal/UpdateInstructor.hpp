////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CS_AUDIO_UPDATE_INSTRUCTOR_HPP
#define CS_AUDIO_UPDATE_INSTRUCTOR_HPP

#include "cs_audio_export.hpp"
// #include "../SourceGroup.hpp"
// #include "../AudioController.hpp"

#include <set>
#include <iostream>
#include <memory>

namespace cs::audio {

class SourceBase;
class SourceGroup;
class AudioController;

class CS_AUDIO_EXPORT UpdateInstructor {
 public:
  UpdateInstructor();
  
  /// @brief Adds a Source to the updateList
  /// @param source Source to add 
  void update(std::shared_ptr<SourceBase> source);

  /// @brief Adds a Source Group, and therefor all Member Sources, to the updateList
  /// @param sourceGroup Source Group to add
  void update(std::shared_ptr<SourceGroup> sourceGroup);
  
  /// @brief Adds an AudioController, and therefor all Sources and Groups 
  /// which live in the controller, to the updateList.
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
    bool updateAll;
    std::shared_ptr<std::vector<std::shared_ptr<SourceBase>>> updateWithGroup = nullptr;
    std::shared_ptr<std::vector<std::shared_ptr<SourceBase>>> updateSourceOnly = nullptr;

    // temporary:
    void print() {
      std::cout << "-----Update Instructions-----" << std::endl;
      std::cout << "updateAll: " << (updateAll ? "true" : "false") << std::endl;
      std::cout << "size group update: " << (updateWithGroup == nullptr ? 0 : updateWithGroup->size()) << std::endl;
      std::cout << "size source update: " << (updateSourceOnly == nullptr ? 0 : updateSourceOnly->size()) << std::endl;
      std::cout << "-----------------------------" << std::endl;
    }
  };

  /// @brief Creates the update instructions when calling AudioController::update().
  /// These UpdateInstructions will contain all sources which need to be updated with their update scope.
  /// There are 3 update scopes: updateAll(When updating the audioController settings. The pipeline will process the audioController and all
  /// Groups and Source on the controller), updateWithGroup(When updating a Group. The pipeline will process all changed groups and all their
  /// members) and updateSourceOnly(When updating a Source. The pipeline will only process the changed source itself). If the updateAll scope is active 
  /// the updateWithGroup and updateSourceOnly Lists get ignored. Otherwise both will be used to determine the sources which need to be updated.
  /// There is a filtering step to ensure that no source is part of both update scopes.
  UpdateInstruction createUpdateInstruction();

 private:                 
  /// List of all source to be updated.
  std::set<std::shared_ptr<SourceBase>>      mSourceUpdateList;
  /// List of all source groups to be updated.
  std::set<std::shared_ptr<SourceGroup>> mGroupUpdateList;
  /// Indicates if the audioController settings changed.
  bool                                   mAudioControllerUpdate;
};

} // namespace cs::audio

#endif // CS_AUDIO_UPDATE_INSTRUCTOR_HPP
