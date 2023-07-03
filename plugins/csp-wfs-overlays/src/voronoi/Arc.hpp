////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSP_WFS_OVERLAYS_ARC_HPP
#define CSP_WFS_OVERLAYS_ARC_HPP

#include "Circle.hpp"
#include "Site.hpp"

namespace csp::wfsoverlays {

class Breakpoint;

struct Arc {
  explicit Arc(Site const& site);

  void invalidateEvent();

  Site        mSite;
  Breakpoint *mLeftBreak, *mRightBreak;

  Circle* mEvent;
};
} // namespace csp::wfsoverlays
#endif // CSP_WFS_OVERLAYS_ARC_HPP
