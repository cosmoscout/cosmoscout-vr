////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CSP_MEASUREMENT_TOOLS_BEACHLINE_HPP
#define CSP_MEASUREMENT_TOOLS_BEACHLINE_HPP

#include "Breakpoint.hpp"
#include "BreakpointTree.hpp"
#include "Site.hpp"

#include <memory>

namespace csp::measurementtools {

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
} // namespace csp::measurementtools
#endif // CSP_MEASUREMENT_TOOLS_BEACHLINE_HPP
