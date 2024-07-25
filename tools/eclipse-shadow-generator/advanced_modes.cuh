////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef ADVANCED_MODES_HPP
#define ADVANCED_MODES_HPP

#include <string>
#include <vector>

// The "advanced" modes compute eclipse shadows for spherical bodies which have an atmosphere. The
// "bruneton" mode uses the Bruneton precomputed atmospheric scattering model to compute the shadow
// map. The required input data is precomputed using the "bruneton-preprocessor" tool of the
// csp-atmospheres plugin. The "planetView" and "atmoView" modes render the atmosphere of a planet
// from the perspective of a given location in the shadow map for debugging and visualization
// purposes.

namespace advanced {

// Computes the shadow map evaluating the Bruneton precomputed atmospheric scattering model for each
// position in the shadow map.
int brunetonMode(std::vector<std::string> const& arguments);

// Draws the atmosphere of a planet into a texture as seen through a pinhole camera. The atmospheric
// scattering data and the position of the observer in shadow map coordinates are given via the
// command line arguments.
int planetViewMode(std::vector<std::string> const& arguments);

// Same as planetMode, but an angular parametrization is used so that the atmosphere fills the
// entire texture regardless of the observer's position.
int atmoViewMode(std::vector<std::string> const& arguments);

} // namespace advanced

#endif // ADVANCED_MODES_HPP