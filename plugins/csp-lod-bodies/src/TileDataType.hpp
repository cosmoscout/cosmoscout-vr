////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

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
