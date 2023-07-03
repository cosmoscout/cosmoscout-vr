////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSP_WFS_OVERLAYS_BEACHLINE_HPP
#define CSP_WFS_OVERLAYS_BEACHLINE_HPP

#include "Breakpoint.hpp"
#include "BreakpointTree.hpp"
#include "Site.hpp"

#include <memory>

namespace csp::wfsoverlays {

class VoronoiGenerator;

class Beachline {
 public:
  explicit Beachline(VoronoiGenerator* parent);

  Arc* insertArcFor(Site const& site);
  void removeArc(Arc* arc);
  void finish(std::vector<Edge>& edges);

 private:
  BreakpointTree    mBreakPoints;
  VoronoiGenerator* mParent;
  Arc*              mRoot;
};
} // namespace csp::wfsoverlays
#endif // CSP_WFS_OVERLAYS_BEACHLINE_HPP
