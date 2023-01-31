////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSP_LOD_BODIES_TILEDATATYPE_HPP
#define CSP_LOD_BODIES_TILEDATATYPE_HPP

#include <iosfwd>

namespace csp::lodbodies {

/// Contains an enumeration of data types that can be stored in a Tile.
enum class TileDataType {
  // Elevation data
  eFloat32 = 0,

  // Image data
  eU8Vec3 = 1
};

std::ostream& operator<<(std::ostream& os, TileDataType tdt);

std::istream& operator>>(std::istream& is, TileDataType& tdt);
} // namespace csp::lodbodies

#endif // CSP_LOD_BODIES_TILEDATATYPE_HPP
