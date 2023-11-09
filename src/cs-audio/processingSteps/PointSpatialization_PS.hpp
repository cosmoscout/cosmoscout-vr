////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CS_AUDIO_PS_POINT_SPATIALIZATION_HPP
#define CS_AUDIO_PS_POINT_SPATIALIZATION_HPP

#include "cs_audio_export.hpp"
#include "ProcessingStep.hpp"
#include "../Source.hpp"
#include <AL/al.h>
#include <glm/fwd.hpp>
#include <chrono>

namespace cs::audio {

class CS_AUDIO_EXPORT PointSpatialization_PS : public ProcessingStep {
 public:
  /// @brief Creates new access to the single PointSpatialization_PS object
  /// @return Pointer to the PS
  static std::shared_ptr<ProcessingStep> create();

  /// @brief processes a source with the given settings
  /// @param source Source to process
  /// @param settings settings to apply
  /// @param failedSettings Pointer to list which contains all failed settings
  void process(std::shared_ptr<Source> source, 
    std::shared_ptr<std::map<std::string, std::any>> settings,
    std::shared_ptr<std::vector<std::string>> failedSettings) override;

  /// @return Wether the processing requires an update call each frame
  bool requiresUpdate() const;

  /// @brief update function to call each frame
  void update();

 private:
  /// Strcut to hold all necessary information regarding a spatialized source
  struct SourceContainer {
    std::weak_ptr<Source> sourcePtr;
    glm::dvec3 currentPos;
    glm::dvec3 lastPos;
  };

  PointSpatialization_PS();
  bool processPosition(std::shared_ptr<Source> source, std::any position);
  /// @brief Calculates and applies the velocity for each spatialized source via the change of position
  void calculateVelocity();

  /// List of all Source which have a position
  std::map<ALuint, SourceContainer> mSourcePositions;
  /// Point in time since the last calculateVelocity() call
  std::chrono::system_clock::time_point mLastTime;
};

} // namespace cs::audio

#endif // CS_AUDIO_PS_POINT_SPATIALIZATION_HPP
