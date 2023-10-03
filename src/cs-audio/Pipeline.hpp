////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CS_AUDIO_PIPELINE_HPP
#define CS_AUDIO_PIPELINE_HPP

#include "cs_audio_export.hpp"
#include "processingsSteps/ProcessingStep.hpp"

#include <vector>
#include <memory>


namespace cs::audio {

class CS_AUDIO_EXPORT Pipeline {
 public:
  bool setPipeline(std::vector<std::shared_ptr<ProcessingStep>> piepline);
  bool appendToPipeline(std::shared_ptr<ProcessingStep> processingStep);
  bool addToPipeline(unsigned int index, std::shared_ptr<ProcessingStep> processingStep);
  bool clearPipeline();
  bool removeFromPipeline(std::shared_ptr<ProcessingStep> processingStep);

  std::vector<std::shared_ptr<ProcessingStep>> getPipeline();

 private:
  std::vector<std::shared_ptr<ProcessingStep>> pipeline;
};

} // namespace cs::audio

#endif // CS_AUDIO_PIPELINE_HPP
