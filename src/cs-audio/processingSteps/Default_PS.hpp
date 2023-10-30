////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CS_AUDIO_PS_DEFAULT_HPP
#define CS_AUDIO_PS_DEFAULT_HPP

#include "cs_audio_export.hpp"
#include "../Source.hpp"
#include "ProcessingStep.hpp"

#include <AL/al.h>

namespace cs::audio {

class CS_AUDIO_EXPORT Default_PS : public ProcessingStep {
 public:

  static std::shared_ptr<ProcessingStep> create();

  void process(ALuint openAlId,
    std::shared_ptr<std::map<std::string, std::any>> settings,
    std::shared_ptr<std::vector<std::string>> failedSettings) override;

  bool requiresUpdate() const;

  void update();

 private:
  Default_PS();
  bool processGain(ALuint openAlId, std::any value);
  bool processLooping(ALuint openAlId, std::any value);
  bool processPitch(ALuint openAlId, std::any value);
};

} // namespace cs::audio

#endif // CS_AUDIO_PS_DEFAULT_HPP
