////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

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
