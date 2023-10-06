////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "Spatialization_PS.hpp"
#include "../internal/alErrorHandling.hpp"
#include "../SourceSettings.hpp"

#include <AL/al.h>
#include <memory>

namespace cs::audio {

void Spatialization_PS::process(ALuint openAlId, std::shared_ptr<SourceSettings> settings) {
    
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cs::audio