////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CS_AUDIO_UPDATE_INSTRUCTOR_HPP
#define CS_AUDIO_UPDATE_INSTRUCTOR_HPP

#include "cs_audio_export.hpp"
#include "../Source.hpp"
#include "../SourceGroup.hpp"
#include "../AudioController.hpp"

#include <set>
#include <memory>

namespace cs::audio {

class CS_AUDIO_EXPORT UpdateInstructor {
 public:
  UpdateInstructor();
  
  /// @brief Adds a source to the updateList
  /// @param source Source to add 
  void update(std::shared_ptr<Source> source);

  /// @brief Adds a source group to the updateList
  /// @param sourceGroup sourceGroup to add
  void update(std::shared_ptr<SourceGroup> sourceGroup);
  
  /// @brief TODO 
  void update(std::shared_ptr<AudioController> audioController);

  /// Struct to hold all update instructions
  struct UpdateInstruction {
    bool updateAll;
    std::shared_ptr<std::vector<std::shared_ptr<Source>>> updateWithGroup = nullptr;
    std::shared_ptr<std::vector<std::shared_ptr<Source>>> updateOnlySource = nullptr;

    // temporary:
    void print() {
      std::cout << "-----Update Instructions-----" << std::endl;
      std::cout << "updateAll: " << (updateAll ? "true" : "false") << std::endl;
      std::cout << "size group update: " << (updateWithGroup == nullptr ? 0 : updateWithGroup->size()) << std::endl;
      std::cout << "size source update: " << (updateOnlySource == nullptr ? 0 : updateOnlySource->size()) << std::endl;
      std::cout << "-----------------------------" << std::endl;
    }
  };

  /// @brief Creates Update instructions for the audioController to 
  /// only call sources that need to be updated within their update scope.
  UpdateInstruction createUpdateInstruction();

 private:                 
  /// List of all source to be updated.
  std::set<std::shared_ptr<Source>>      mSourceUpdateList;
  /// List of all source groups to be updated.
  std::set<std::shared_ptr<SourceGroup>> mGroupUpdateList;
  /// Indicates if the plugin settings changed.
  bool                                   mPluginUpdate;
};

} // namespace cs::audio

#endif // CS_AUDIO_UPDATE_INSTRUCTOR_HPP
