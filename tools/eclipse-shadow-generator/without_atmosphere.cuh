////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef WITHOUT_ATMOSPHERE_HPP
#define WITHOUT_ATMOSPHERE_HPP

#include "LimbDarkening.cuh"
#include "types.hpp"

// Computes the shadow map by sampling the intersection area between circles representing the Sun
// and the occluder. This makes use of the global limb darkening function.
__global__ void computeLimbDarkeningShadow(float* shadowMap, ShadowSettings settings, LimbDarkening limbDarkening);

// Computes the shadow map by analytically computing the intersection area between circles
// representing the Sun and the occluder. This does not use a limb darkening function.
__global__ void computeCircleIntersectionShadow(float* shadowMap, ShadowSettings settings);

// Computes the shadow map by assuming a linear brightness gradient from the outer edge of the
// penumbra to the start of the umbra / antumbra. In the antumbra, the shadow intensity decreases
// quadratically. This does not use a limb darkening function.
__global__ void computeLinearShadow(float* shadowMap, ShadowSettings settings);

// Computes the shadow map by assuming a smoothstep-based brightness gradient from the outer edge of
// the penumbra to the start of the umbra / antumbra. In the antumbra, the shadow intensity
// decreases quadratically. This does not use a limb darkening function.
__global__ void computeSmoothstepShadow(float* shadowMap, ShadowSettings settings);


#endif // WITHOUT_ATMOSPHERE_HPP