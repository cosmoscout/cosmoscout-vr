////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CS_AUDIO_PS_POINT_SPATIALIZATION_HPP
#define CS_AUDIO_PS_POINT_SPATIALIZATION_HPP

#include "../internal/SourceBase.hpp"
#include "ProcessingStep.hpp"
#include "SpatializationUtils.hpp"
#include "cs_audio_export.hpp"
#include <AL/al.h>
#include <chrono>
#include <glm/fwd.hpp>

namespace cs::audio {
/*
The PointSpatialization_PS is a spatialization processing step with which lets you define a position
as a single point in space. This processing step will also automatically compute the velocity of a
source and the observer. The position must be specified relative to the observer.
---------------------------------------------------------
Name        Type          Range       Description
---------------------------------------------------------
position    glm::dvec3                Position of a source relative to the observer.
---------------------------------------------------------
*/
class CS_AUDIO_EXPORT PointSpatialization_PS : public ProcessingStep, public SpatializationUtils {
 public:
  /// @brief Creates new access to the single PointSpatialization_PS object
  /// @return Pointer to the PS
  static std::shared_ptr<ProcessingStep> create(bool stationaryOutputDevice);

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
  PointSpatialization_PS(bool stationaryOutputDevice);
  bool processPosition(std::shared_ptr<SourceBase> source, std::any position);
};

} // namespace cs::audio

#endif // CS_AUDIO_PS_POINT_SPATIALIZATION_HPP
