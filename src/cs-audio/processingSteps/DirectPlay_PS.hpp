////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CS_AUDIO_PS_DIRECT_PLAY_HPP
#define CS_AUDIO_PS_DIRECT_PLAY_HPP

#include "../internal/SourceBase.hpp"
#include "ProcessingStep.hpp"
#include "cs_audio_export.hpp"

#include <AL/al.h>

namespace cs::audio {

/*
DirectPlay_PS is the most basic playback control processing step. It will immediately apply
the specified playback setting.
As with all playback control processing steps the playback setting can be set via the play(),
pause() and stop() functions of a source.
--------------------------------------------
Name      Type          Range     Description
--------------------------------------------
playback  std::string   "play"    playback option
                        "stop"
                        "pause"
--------------------------------------------
*/
class CS_AUDIO_EXPORT DirectPlay_PS : public ProcessingStep {
 public:
  /// @brief Creates new access to the single DirectPlay_PS object
  /// @return Pointer to the PS
  static std::shared_ptr<ProcessingStep> create();

  /// @brief processes a source with the given settings
  /// @param source Source to process
  /// @param settings settings to apply
  /// @param failedSettings Pointer to list which contains all failed settings
  void process(std::shared_ptr<SourceBase>             source,
      std::shared_ptr<std::map<std::string, std::any>> settings,
      std::shared_ptr<std::vector<std::string>>        failedSettings) override;

  /// @return Wether the processing requires an update call each frame
  bool requiresUpdate() const override;

  /// @brief update function to call each frame
  void update() override;

 private:
  DirectPlay_PS();
  bool processPlayback(ALuint openAlId, std::any value);
};

} // namespace cs::audio

#endif // CS_AUDIO_PS_DIRECT_PLAY_HPP
