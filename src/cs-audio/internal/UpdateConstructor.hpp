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

 private:
  UpdateConstructor(std::shared_ptr<ProcessingStepsManager> processingStepsManager);

  std::shared_ptr<ProcessingStepsManager> mProcessingStepsManager;         
};

} // namespace cs::audio

#endif // CS_AUDIO_UPDATE_CONSTRUCTOR_HPP
