////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSP_MEASUREMENT_TOOLS_ARC_HPP
#define CSP_MEASUREMENT_TOOLS_ARC_HPP

#include "Circle.hpp"
#include "Site.hpp"

namespace csp::measurementtools {

class Breakpoint;

struct Arc {
  explicit Arc(Site const& site);

  void invalidateEvent();

  Site        mSite;
  Breakpoint *mLeftBreak, *mRightBreak;

  Circle* mEvent;
};
} // namespace csp::measurementtools
#endif // CSP_MEASUREMENT_TOOLS_ARC_HPP
