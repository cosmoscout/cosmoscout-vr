////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef SIMPLE_MODES_HPP
#define SIMPLE_MODES_HPP

#include <string>
#include <vector>

// The "simple" modes compute eclipse shadows for spherical bodies which do not have an atmosphere.
// The most accurate mode is "limbDarkeningMode", which uses a limb darkening function to model the
// surface brightness of the Sun. The other modes are less accurate, and are mostly provided for
// comparison purposes. More details can be found in the paper "Real-Time Rendering of Eclipses
// without Incorporation of Atmospheric Effects"
// (https://onlinelibrary.wiley.com/doi/full/10.1111/cgf.14676).

namespace simple {

// Computes the shadow map by sampling a limb-darkening model in the intersection area between
// circles representing the Sun and the occluder.
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

#endif // SIMPLE_MODES_HPP