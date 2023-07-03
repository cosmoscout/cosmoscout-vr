////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSP_WFS_OVERLAYS_CIRCLE_HPP
#define CSP_WFS_OVERLAYS_CIRCLE_HPP

#include "Site.hpp"
#include "Vector2f.hpp"

namespace csp::wfsoverlays {

struct Arc;
struct Site;

struct Circle {
  Circle(Arc* arc, double sweepLine);

  Site     mSite;
  Vector2f mCenter;
  Arc*     mArc;
  bool     mIsValid;
  Vector2f mPriority;
};

bool operator<(Circle const& lhs, Circle const& rhs);

struct CirclePtrCmp {
  bool operator()(const Circle* lhs, const Circle* rhs) const {
    return *lhs < *rhs;
  }
};
} // namespace csp::wfsoverlays
#endif // CSP_WFS_OVERLAYS_CIRCLE_HPP
