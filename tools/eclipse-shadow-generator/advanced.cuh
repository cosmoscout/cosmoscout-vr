////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef ADVANCED_HPP
#define ADVANCED_HPP

#include <string>
#include <vector>

namespace advanced {

// Draws the atmosphere of a planet into a texture as seen through a pinhole camera. The atmospheric
// scattering data, the position of the observer relative to the planet and the sun's direction are
// given via command line arguments.
int planetViewMode(std::vector<std::string> const& arguments);

// Same as planetMode, but an angular parametrization is used so that the atmosphere fills the
// entire texture regardless of the observer's position.
int atmoViewMode(std::vector<std::string> const& arguments);

} // namespace advanced

#endif // ADVANCED_HPP