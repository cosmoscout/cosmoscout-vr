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
  // scalar types
  eFloat32 = 0,
  eUInt8   = 1,

  // vector types
  eU8Vec3 = 2
};

std::ostream& operator<<(std::ostream& os, TileDataType tdt);

std::istream& operator>>(std::istream& is, TileDataType& tdt);
} // namespace csp::lodbodies

#endif // CSP_LOD_BODIES_TILEDATATYPE_HPP
