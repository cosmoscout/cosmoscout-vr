////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CS_AUDIO_UPDATE_CONSTRUCTOR_HPP
#define CS_AUDIO_UPDATE_CONSTRUCTOR_HPP

#include "cs_audio_export.hpp"

#include <vector>
#include <memory>
#include <map>
#include <any>
#include <string>

namespace cs::audio {

class SourceBase;
class SourceGroup;
class AudioController;
class ProcessingStepsManager;

class CS_AUDIO_EXPORT UpdateConstructor {
 public:
  static std::shared_ptr<UpdateConstructor> createUpdateConstructor(
    std::shared_ptr<ProcessingStepsManager> processingStepsManager);
  
  void updateAll(
    std::shared_ptr<std::vector<std::shared_ptr<SourceBase>>> sources, 
    std::shared_ptr<std::vector<std::shared_ptr<SourceGroup>>> groups,
    std::shared_ptr<AudioController> audioController);
  void updateGroups(
    std::shared_ptr<std::vector<std::shared_ptr<SourceBase>>> sources, 
    std::shared_ptr<std::vector<std::shared_ptr<SourceGroup>>> groups,
    std::shared_ptr<AudioController> audioController);
  void updateSources(
    std::shared_ptr<std::vector<std::shared_ptr<SourceBase>>> sources,
    std::shared_ptr<AudioController> audioController);
    
  /// @brief Update source settings with the currently set settings of the audio Controller.
  /// Is only ever called when a new source gets created.
  /// @param source source to update
  /// @param audioController audioController in which the source lives
  /// @param settings audio controller settings to apply to source
  void applyCurrentControllerSettings(
    std::shared_ptr<SourceBase> source,
    std::shared_ptr<AudioController> audioController,
    std::shared_ptr<std::map<std::string, std::any>> settings);

  /// @brief Update source settings with the currently set settings of a group.
  /// Is only ever called when a source gets added to a group.
  /// @param source source to update
  /// @param audioController audioController in which the source lives
  /// @param settings group settings to apply to source
  void applyCurrentGroupSettings(
    std::shared_ptr<SourceBase> source,
    std::shared_ptr<AudioController> audioController,
    std::shared_ptr<std::map<std::string, std::any>> settings);

 private:
  UpdateConstructor(std::shared_ptr<ProcessingStepsManager> processingStepsManager);
  
  bool containsRemove(std::shared_ptr<std::map<std::string, std::any>> settings);
  void rebuildPlaybackSettings(std::shared_ptr<AudioController> audioController, std::shared_ptr<SourceBase> source);

  std::shared_ptr<ProcessingStepsManager> mProcessingStepsManager;         
};

} // namespace cs::audio

#endif // CS_AUDIO_UPDATE_CONSTRUCTOR_HPP
