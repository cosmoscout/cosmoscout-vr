////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CS_AUDIO_UPDATE_BUILDER_HPP
#define CS_AUDIO_UPDATE_BUILDER_HPP

#include "cs_audio_export.hpp"
#include "../Source.hpp"
#include "../SourceGroup.hpp"

#include <set>
#include <memory>

namespace cs::audio {

class CS_AUDIO_EXPORT UpdateBuilder {
 public:
  UpdateBuilder();
  
  /// @brief Adds a source to the updateList
  /// @param source Source to add 
  void update(Source* source);

  /// @brief Adds a source group to the updateList
  /// @param sourceGroup sourceGroup to add
  void update(SourceGroup* sourceGroup);
  
  /// @brief TODO 
  void updatePlugin();

  /// Struct to hold all update instructions
  struct UpdateList {
    bool updateAll;
    std::vector<std::shared_ptr<Source>> updateWithGroup;
    std::vector<std::shared_ptr<Source>> updateOnlySource;

    // temporary:
    void print() {
      std::cout << "-----Update Instructions-----" << std::endl;
      std::cout << "updateAll: " << (updateAll ? "true" : "false") << std::endl;
      std::cout << "size group update: " << updateWithGroup.size() << std::endl;
      std::cout << "size source update: " << updateOnlySource.size() << std::endl;
      std::cout << "-----------------------------" << std::endl;
    }
  };

  /// @brief Creates Update instructions for the audioController to 
  /// only call sources that need to be updated within their update scope.
  /// @return Update instructions.
  UpdateList createUpdateList();

 private:                 
  /// List of all source to be updated.
  std::set<std::shared_ptr<Source>>      mSourceUpdateList;
  /// List of all source groups to be updated.
  std::set<std::shared_ptr<SourceGroup>> mGroupUpdateList;
  /// Indicates if the plugin settings changed.
  bool                                   mPluginUpdate;
};

} // namespace cs::audio

#endif // CS_AUDIO_UPDATE_BUILDER_HPP
