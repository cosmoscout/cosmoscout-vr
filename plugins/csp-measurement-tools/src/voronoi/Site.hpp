////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSP_MEASUREMENT_TOOLS_SITE_HPP
#define CSP_MEASUREMENT_TOOLS_SITE_HPP

#include <cstdint>
namespace csp::measurementtools {

struct Site {
  Site(double x, double y, uint16_t a = 0);

  double mX;
  double mY;

  uint16_t mAddr;
};

bool operator<(Site const& lhs, Site const& rhs);

bool operator==(Site const& lhs, Site const& rhs);

struct SitePosComp {
  bool operator()(Site const& lhs, Site const& rhs) const {
    return (lhs.mY == rhs.mY) ? (lhs.mX < rhs.mX) : (lhs.mY > rhs.mY);
  }
};
} // namespace csp::measurementtools
#endif // CSP_MEASUREMENT_TOOLS_SITE_HPP
