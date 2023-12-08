////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CS_AUDIO_PS_DEFAULT_HPP
#define CS_AUDIO_PS_DEFAULT_HPP

#include "../internal/SourceBase.hpp"
#include "ProcessingStep.hpp"
#include "cs_audio_export.hpp"

#include <AL/al.h>

namespace cs::audio {

/*
This Processing Step introduces basic settings for a source and is automatically enabled
in every pipeline.
--------------------------------------------
Name      Type    Range      Description
--------------------------------------------
gain      float   0.0 -      Multiplier for the Volume of a source
pitch     float   0.5 - 2.0  Multiplier for the sample rate of the source's buffer
looping   bool               Whether the source shall loop the playback or stop after
                             playing the buffer once.
--------------------------------------------
*/
class CS_AUDIO_EXPORT Default_PS : public ProcessingStep {
 public:
  /// @brief Creates new access to the single Default_PS object
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
  Default_PS();
  bool processGain(ALuint openAlId, std::any value);
  bool processLooping(std::shared_ptr<SourceBase> source, std::any value);
  bool processPitch(ALuint openAlId, std::any value);
};

} // namespace cs::audio

#endif // CS_AUDIO_PS_DEFAULT_HPP
