////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "Spatialization_PS.hpp"
#include "../internal/alErrorHandling.hpp"

#include <AL/al.h>
#include <memory>
#include <vector>
#include <string>

namespace cs::audio {

std::shared_ptr<ProcessingStep> Spatialization_PS::create() {
  static auto spatialization_ps = std::shared_ptr<Spatialization_PS>(new Spatialization_PS());
  return spatialization_ps;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

Spatialization_PS::Spatialization_PS() {}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Spatialization_PS::process(ALuint openAlId, 
  std::shared_ptr<std::map<std::string, std::any>> settings,
  std::shared_ptr<std::vector<std::string>> failedSettings) {
    
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cs::audio