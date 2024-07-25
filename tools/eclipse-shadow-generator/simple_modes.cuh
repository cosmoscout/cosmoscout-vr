////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef SIMPLE_HPP
#define SIMPLE_HPP

#include <string>
#include <vector>

namespace simple {

// Computes the shadow map by sampling the intersection area between circles representing the Sun
// and the occluder. This makes use of the global limb darkening function.
int limbDarkeningMode(std::vector<std::string> const& arguments);

// Computes the shadow map by analytically computing the intersection area between circles
// representing the Sun and the occluder. This does not use a limb darkening function.
int circleIntersectionMode(std::vector<std::string> const& arguments);

// Computes the shadow map by assuming a linear brightness gradient from the outer edge of the
// penumbra to the start of the umbra / antumbra. In the antumbra, the shadow intensity decreases
// quadratically. This does not use a limb darkening function.
int linearMode(std::vector<std::string> const& arguments);

// Computes the shadow map by assuming a smoothstep-based brightness gradient from the outer edge of
// the penumbra to the start of the umbra / antumbra. In the antumbra, the shadow intensity
// decreases quadratically. This does not use a limb darkening function.
int smoothstepMode(std::vector<std::string> const& arguments);

} // namespace simple

#endif // SIMPLE_HPP