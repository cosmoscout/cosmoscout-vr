////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CS_AUDIO_PS_DISTANCE_CULLING_HPP
#define CS_AUDIO_PS_DISTANCE_CULLING_HPP

#include "cs_audio_export.hpp"
#include "ProcessingStep.hpp"
#include "../internal/SourceBase.hpp"

#include <AL/al.h>

namespace cs::audio {

class CS_AUDIO_EXPORT DistanceCulling_PS : public ProcessingStep {
 public:
  /// @brief Creates new access to the single DistanceCulling_PS object
  /// @return Pointer to the PS
  static std::shared_ptr<ProcessingStep> create(double distanceThreshold);

  /// @brief processes a source with the given settings
  /// @param source Source to process
  /// @param settings settings to apply
  /// @param failedSettings Pointer to list which contains all failed settings
  void process(std::shared_ptr<SourceBase> source, 
    std::shared_ptr<std::map<std::string, std::any>> settings,
    std::shared_ptr<std::vector<std::string>> failedSettings) override;

  /// @return Wether the processing requires an update call each frame
  bool requiresUpdate() const;

  /// @brief update function to call each frame
  void update();

 private:
  double mDistanceThreshold;

  DistanceCulling_PS(double distanceThreshold);
  bool processPosition(std::shared_ptr<SourceBase>, std::any position, std::any newPlayback);
};

} // namespace cs::audio

#endif // CS_AUDIO_PS_DISTANCE_CULLING_HPP
