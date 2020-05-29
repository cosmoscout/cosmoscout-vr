////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "TileDataType.hpp"

#include <fstream>
#include <istream>
#include <sstream>
#include <string>

namespace csp::lodbodies {

////////////////////////////////////////////////////////////////////////////////////////////////////

std::ostream& operator<<(std::ostream& os, TileDataType tdt) {
  switch (tdt) {
  case TileDataType::eFloat32:
    os << "Float32";
    break;

  case TileDataType::eUInt8:
    os << "UInt8";
    break;

  case TileDataType::eU8Vec3:
    os << "U8Vec3";
    break;

    // no default - to get compiler warning when the set of enum values is
    // extended.
  };

  return os;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::istream& operator>>(std::istream& is, TileDataType& tdt) {
  std::string s;
  is >> s;

  if (s == "Float32") {
    tdt = TileDataType::eFloat32;
  } else if (s == "UInt8") {
    tdt = TileDataType::eUInt8;
  } else if (s == "U8Vec3") {
    tdt = TileDataType::eU8Vec3;
  }

  return is;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::lodbodies
