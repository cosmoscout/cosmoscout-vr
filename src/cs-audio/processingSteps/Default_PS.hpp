////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CS_AUDIO_PS_DEFAULT_HPP
#define CS_AUDIO_PS_DEFAULT_HPP

#include "cs_audio_export.hpp"
#include "../internal/OpenAlError.hpp"
#include "../SourceSettings.hpp"
#include "../Source.hpp"
#include "ProcessingStep.hpp"

#include <AL/al.h>

namespace cs::audio {

class CS_AUDIO_EXPORT Default_PS : public OpenAlError, public ProcessingStep {
 public:
 void process(ALuint openAlId);

 private:
};

} // namespace cs::audio

#endif // CS_AUDIO_PS_DEFAULT_HPP
