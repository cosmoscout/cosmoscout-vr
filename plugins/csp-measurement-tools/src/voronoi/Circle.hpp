////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CSP_MEASUREMENT_TOOLS_CIRCLE_HPP
#define CSP_MEASUREMENT_TOOLS_CIRCLE_HPP

#include "Site.hpp"
#include "Vector2f.hpp"

namespace csp::measurementtools {

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
} // namespace csp::measurementtools
#endif // CSP_MEASUREMENT_TOOLS_CIRCLE_HPP
