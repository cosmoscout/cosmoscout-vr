////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

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
