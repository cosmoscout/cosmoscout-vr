////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CSP_LOD_BODIES_TILEBOUNDS_HPP
#define CSP_LOD_BODIES_TILEBOUNDS_HPP

#include "BoundingBox.hpp"
#include "Tile.hpp"

namespace csp::lodbodies {

/// Returns the bounds of tile for a planet of the given equatorial radius radiusE, polar radius
/// radiusP, and heightScale.
///
/// Assumes that tile stores elevation data (i.e. a single scalar) and has valid min/max values set.
BoundingBox<double> calcTileBounds(
    TileBase const& tile, double radiusE = 1.F, double radiusP = 1.F, double heightScale = 1.F);

BoundingBox<double> calcTileBounds(double tmin, double tmax, int tileLevel, glm::int64 patchIdx,
    double radiusE = 1.F, double radiusP = 1.F, double heightScale = 1.F);
} // namespace csp::lodbodies

#endif // CSP_LOD_BODIES_TILEBOUNDS_HPP
