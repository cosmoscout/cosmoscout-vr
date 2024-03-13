////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CS_AUDIO_UPDATE_CONSTRUCTOR_HPP
#define CS_AUDIO_UPDATE_CONSTRUCTOR_HPP

#include "cs_audio_export.hpp"
#include <any>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace cs::audio {

// forward declaration
class SourceBase;
class SourceGroup;
class AudioController;
class ProcessingStepsManager;

/// @brief This class takes the controller, groups and sources, which need to be updated, and
/// builds the final settings taking the hierarchy of these classes into account.
/// The general rule is 'local overwrites global', meaning the hierarchy looks like this:
/// controller < group < source, where source has the highest level. The idea is that the final
/// settings only contain things that are changing. Once the settings are computed, the
/// UpdateConstructor will call the pipeline and write the currently set settings to the controller,
/// groups, sources and playbackSettings. This class should only be instantiated once.
class CS_AUDIO_EXPORT UpdateConstructor {
 public:
  UpdateConstructor(std::shared_ptr<ProcessingStepsManager> processingStepsManager);
  ~UpdateConstructor();

  /// @brief Updates the controller, all groups and all sources
  /// @param sources sources to update
  /// @param groups groups to update
  /// @param audioController controller to update
  void updateAll(std::shared_ptr<std::vector<std::shared_ptr<SourceBase>>> sources,
      std::shared_ptr<std::vector<std::shared_ptr<SourceGroup>>>           groups,
      std::shared_ptr<AudioController>                                     audioController);

  /// @brief Updates the groups and all sources
  /// @param sources sources to update
  /// @param groups groups to update
  /// @param audioController audio controller on which sources and groups live
  void updateGroups(std::shared_ptr<std::vector<std::shared_ptr<SourceBase>>> sources,
      std::shared_ptr<std::vector<std::shared_ptr<SourceGroup>>>              groups,
      std::shared_ptr<AudioController>                                        audioController);

  /// @brief Update all sources
  /// @param sources sources to update
  /// @param audioController audio controller on which the source lives
  void updateSources(std::shared_ptr<std::vector<std::shared_ptr<SourceBase>>> sources,
      std::shared_ptr<AudioController>                                         audioController);

  /// @brief Update source settings with the currently set settings of the audio Controller.
  /// Is only ever called when a new source gets created.
  /// @param source source to update
  /// @param audioController audioController on which the source lives
  /// @param settings audio controller settings to apply to source
  void applyCurrentControllerSettings(std::shared_ptr<SourceBase> source,
      std::shared_ptr<AudioController>                            audioController,
      std::shared_ptr<std::map<std::string, std::any>>            settings);

  /// @brief Update source settings with the currently set settings of a group.
  /// Is only ever called when a source gets added to a group.
  /// @param source source to update
  /// @param audioController audioController on which the source lives
  /// @param settings group settings to apply to source
  void applyCurrentGroupSettings(std::shared_ptr<SourceBase> source,
      std::shared_ptr<AudioController>                       audioController,
      std::shared_ptr<std::map<std::string, std::any>>       settings);

  /// @brief Update source settings and remove the currently set settings of a group.
  /// Gets called when a source leaves a group or the group gets deleted.
  /// @param source source to update
  /// @param audioController audioController on which the source lives
  void removeCurrentGroupSettings(
      std::shared_ptr<SourceBase> source, std::shared_ptr<AudioController> audioController);

 private:
  /// @brief Checks whether a settings map contains a remove setting
  /// @param settings settings to check
  /// @return True if settings contains remove
  bool containsRemove(std::shared_ptr<std::map<std::string, std::any>> settings);

  /// @brief Completely rebuilds the settings of source by taking the current and update setting
  /// of the source, group and controller into account. This is only done if there is at least
  /// one settings that gets removed. This is needed because if a setting gets removed, the
  /// hierarchy can reverse, meaning a lower level could overwrite a higher one. Completely
  /// rebuilding it is easier instead of trying to figure out if and how a lower level one could
  /// overwrite a higher one.
  /// @param audioController audioController on which the source lives.
  /// @param source source to update
  void rebuildPlaybackSettings(
      std::shared_ptr<AudioController> audioController, std::shared_ptr<SourceBase> source);

  std::shared_ptr<ProcessingStepsManager> mProcessingStepsManager;
};

} // namespace cs::audio

#endif // CS_AUDIO_UPDATE_CONSTRUCTOR_HPP
