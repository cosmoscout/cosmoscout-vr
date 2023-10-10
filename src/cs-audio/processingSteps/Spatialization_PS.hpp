////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CS_AUDIO_PS_SPATIALIZATION_HPP
#define CS_AUDIO_PS_SPATIALIZATION_HPP

#include "cs_audio_export.hpp"
#include "ProcessingStep.hpp"

#include <AL/al.h>

namespace cs::audio {

class CS_AUDIO_EXPORT Spatialization_PS : public ProcessingStep {
 public:
  void process(ALuint openAlId, std::shared_ptr<std::map<std::string, std::any>> settings);

 private:
};

} // namespace cs::audio

#endif // CS_AUDIO_PS_SPATIALIZATION_HPP
