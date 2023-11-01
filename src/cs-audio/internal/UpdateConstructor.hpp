////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CS_AUDIO_UPDATE_CONSTRUCTOR_HPP
#define CS_AUDIO_UPDATE_CONSTRUCTOR_HPP

#include "cs_audio_export.hpp"
// #include "../Source.hpp"
// #include "../SourceGroup.hpp"
// #include "../AudioController.hpp"

#include <vector>
#include <memory>
#include <map>
#include <any>
#include <string>

namespace cs::audio {

class Source;
class SourceGroup;
class AudioController;
class ProcessingStepsManager;

class CS_AUDIO_EXPORT UpdateConstructor {
 public:
  static std::shared_ptr<UpdateConstructor> createUpdateConstructor(
    std::shared_ptr<ProcessingStepsManager> processingStepsManager);

  void updateAll(
    std::shared_ptr<std::vector<std::shared_ptr<Source>>> sources, 
    std::shared_ptr<std::vector<std::shared_ptr<SourceGroup>>> groups,
    AudioController* audioController);
  void updateGroups(
    std::shared_ptr<std::vector<std::shared_ptr<Source>>> sources, 
    std::shared_ptr<std::vector<std::shared_ptr<SourceGroup>>> groups,
    AudioController* audioController);
  void updateSources(
    std::shared_ptr<std::vector<std::shared_ptr<Source>>> sources,
    AudioController* audioController);
    
  /// @brief Update source settings with the currently set settings of the audio Controller.
  /// Is only called whenever a new source gets created.
  /// @param source source to update
  /// @param audioController audioController in which the source lives
  /// @param settings audio controller settings to apply to source
  void applyCurrentControllerSettings(
    std::shared_ptr<Source> source,
    AudioController* audioController,
    std::shared_ptr<std::map<std::string, std::any>> settings);

  /// @brief Update source settings with the currently set settings of a group.
  /// Is called whenever a source gets added to a group.
  /// @param source source to update
  /// @param audioController audioController in which the source lives
  /// @param group group settings to apply to source
  void applyCurrentGroupSettings(
    std::shared_ptr<Source> source,
    AudioController* audioController,
    std::shared_ptr<std::map<std::string, std::any>> settings);

 private:
  UpdateConstructor(std::shared_ptr<ProcessingStepsManager> processingStepsManager);
  void eraseRemoveSettings(std::shared_ptr<std::map<std::string, std::any>> settings);

  std::shared_ptr<ProcessingStepsManager> mProcessingStepsManager;         
};

} // namespace cs::audio

#endif // CS_AUDIO_UPDATE_CONSTRUCTOR_HPP
