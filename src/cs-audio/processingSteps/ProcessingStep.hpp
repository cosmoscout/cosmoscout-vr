////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CS_AUDIO_PROCESSING_STEP_HPP
#define CS_AUDIO_PROCESSING_STEP_HPP

#include "cs_audio_export.hpp"

#include <AL/al.h>
#include <map>
#include <any>

namespace cs::audio {

class CS_AUDIO_EXPORT ProcessingStep {
 public:
  virtual void process(ALuint openAlId, std::shared_ptr<std::map<std::string, std::any>> settings) = 0;
 
 private:

};

} // namespace cs::audio

#endif // CS_AUDIO_PROCESSING_STEP_HPP
