////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CS_AUDIO_PS_VOLUME_CULLING_HPP
#define CS_AUDIO_PS_VOLUME_CULLING_HPP

#include "../internal/SourceBase.hpp"
#include "ProcessingStep.hpp"
#include "cs_audio_export.hpp"

#include <AL/al.h>

namespace cs::audio {

/*
VolumeCulling_PS is a playback control processing step. If the playback option is set to "play" it
will play a source if it's theoretical volume is greater then the specified volume culling
threshold. This theoretical volume is calculated according to a sources distance model formula and
multiplied by the set gain via Default_PS. This volume does not necessarily reflect the actual
volume of a source because there many more factors that can have an influence. This processing step
will only get active if a source has a postion. If this is not the case the source will never get
played. As with all playback control processing steps the playback setting can be set via the
play(), pause() and stop() functions of a source.
--------------------------------------------
Name      Type          Range     Description
--------------------------------------------
playback  std::string   "play"    playback option
                        "stop"
                        "pause"
--------------------------------------------
*/
class CS_AUDIO_EXPORT VolumeCulling_PS : public ProcessingStep {
 public:
  /// @brief Creates new access to the single Default_PS object
  /// @return Pointer to the PS
  static std::shared_ptr<ProcessingStep> create(float gainThreshold);

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
  float mGainThreshold;

  VolumeCulling_PS(float gainThreshold);
  bool processPosition(
      std::shared_ptr<SourceBase>, std::any position, std::any newGain, std::any newPlayback);
  double inverseClamped(
      double distance, ALfloat rollOffFactor, ALfloat referenceDistance, ALfloat maxDistance) const;
  double linearClamped(
      double distance, ALfloat rollOffFactor, ALfloat referenceDistance, ALfloat maxDistance) const;
  double exponentClamped(
      double distance, ALfloat rollOffFactor, ALfloat referenceDistance, ALfloat maxDistance) const;
};

} // namespace cs::audio

#endif // CS_AUDIO_PS_VOLUME_CULLING_HPP
