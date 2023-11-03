////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "TileBounds.hpp"

#include "../../../src/cs-utils/convert.hpp"
#include "HEALPix.hpp"

namespace csp::lodbodies {

////////////////////////////////////////////////////////////////////////////////////////////////////

BoundingBox<double> calcTileBounds(double tmin, double tmax, int tileLevel, glm::int64 patchIdx,
    glm::dvec3 const& radii, double heightScale) {
  BoundingBox<double> result;

  // min/max elevation of tile, adjusted for heightScale
  double const        tMin = heightScale * tmin;
  double const        tMax = heightScale * tmax;
  HEALPixLevel const& hp   = HEALPix::getLevel(tileLevel);

  // min/max corner of the tile's bounding box
  glm::dvec3 bbMin(std::numeric_limits<double>::max(), std::numeric_limits<double>::max(),
      std::numeric_limits<double>::max());
  glm::dvec3 bbMax(std::numeric_limits<double>::lowest(), std::numeric_limits<double>::lowest(),
      std::numeric_limits<double>::lowest());

  // To obtain corners of the bounding box calculate the position of the
  // surface in the 4 corners, center of the tile, and center of the edges
  // as if the surface elevation takes the min/max value at those locations.
  // Then take componentwise min/max to get the axis aligned box.
  glm::dvec2 center = hp.getCenterLngLat(patchIdx);

  // tile center
  {
    // lowest/highest point at the center
    glm::dvec3 pMin(cs::utils::convert::toCartesian(center, radii, tMin));
    glm::dvec3 pMax(cs::utils::convert::toCartesian(center, radii, tMax));

    bbMin = glm::min(bbMin, pMin);
    bbMin = glm::min(bbMin, pMax);
    bbMax = glm::max(bbMax, pMin);
    bbMax = glm::max(bbMax, pMax);
  }

  // tile corners
  auto corners = hp.getCornersLngLat(patchIdx);

  for (auto const& corner : corners) {
    // lowest/highest point at corner
    glm::dvec3 pMin(cs::utils::convert::toCartesian(corner, radii, tMin));
    glm::dvec3 pMax(cs::utils::convert::toCartesian(corner, radii, tMax));

    bbMin = glm::min(bbMin, pMin);
    bbMin = glm::min(bbMin, pMax);
    bbMax = glm::max(bbMax, pMin);
    bbMax = glm::max(bbMax, pMax);
  }

  // tile edge center points
  auto edges = hp.getEdgeCentersLngLat(patchIdx);

  for (auto const& edge : edges) {
    // lowest/highest point at edge center
    glm::dvec3 pMin(cs::utils::convert::toCartesian(edge, radii, tMin));
    glm::dvec3 pMax(cs::utils::convert::toCartesian(edge, radii, tMax));

    bbMin = glm::min(bbMin, pMin);
    bbMin = glm::min(bbMin, pMax);
    bbMax = glm::max(bbMax, pMin);
    bbMax = glm::max(bbMax, pMax);
  }

  // store results
  result.setMin(bbMin);
  result.setMax(bbMax);

  return result;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Returns the bounds of @a tile for a planet of the given equatorial radius
// @a radiusE and polar radius @a radiusP
// @a heightScale.
// @note Assumes that @a tile stores elevation data (i.e. a single scalar) and
// has valid min/max values set.
BoundingBox<double> calcTileBounds(
    TileNode const& tile, glm::dvec3 const& radii, double heightScale) {

  if (tile.getMinMaxPyramid()) {
    return calcTileBounds(tile.getMinMaxPyramid()->getMin(), tile.getMinMaxPyramid()->getMax(),
        tile.getLevel(), tile.getPatchIdx(), radii, heightScale);
  }

  return BoundingBox<double>();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::lodbodies
