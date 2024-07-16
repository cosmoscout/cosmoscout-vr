////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef WITH_ATMOSPHERE_HPP
#define WITH_ATMOSPHERE_HPP

#include "LimbDarkening.cuh"
#include "types.hpp"

#include <string>

void computeAtmosphereShadow(float* shadowMap, ShadowSettings settings, std::string const& atmosphereSettings, LimbDarkening limbDarkening);

#endif // WITH_ATMOSPHERE_HPP