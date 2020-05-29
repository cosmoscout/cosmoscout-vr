////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "Site.hpp"
#include <cstdint>

namespace csp::measurementtools {

Site::Site(double x_in, double y_in, uint16_t a)
    : mX(x_in)
    , mY(y_in)
    , mAddr(a) {
}

bool operator<(Site const& lhs, Site const& rhs) {
  return lhs.mAddr < rhs.mAddr;
}

bool operator==(Site const& lhs, Site const& rhs) {
  return lhs.mAddr == rhs.mAddr;
}
} // namespace csp::measurementtools
