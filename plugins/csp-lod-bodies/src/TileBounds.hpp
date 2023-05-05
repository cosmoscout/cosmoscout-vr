////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSP_LOD_BODIES_TILEBOUNDS_HPP
#define CSP_LOD_BODIES_TILEBOUNDS_HPP

#include "BoundingBox.hpp"
#include "TileData.hpp"

namespace csp::lodbodies {

/// Returns the bounds of tile for a planet of the given equatorial radius radiusE, polar radius
/// radiusP, and heightScale.
///
/// Assumes that tile stores elevation data (i.e. a single scalar) and has valid min/max values set.
BoundingBox<double> calcTileBounds(
    TileBase const& tile, glm::dvec3 const& radii, double heightScale);

BoundingBox<double> calcTileBounds(double tmin, double tmax, int tileLevel, glm::int64 patchIdx,
    glm::dvec3 const& radii, double heightScale);
} // namespace csp::lodbodies

#endif // CSP_LOD_BODIES_TILEBOUNDS_HPP
