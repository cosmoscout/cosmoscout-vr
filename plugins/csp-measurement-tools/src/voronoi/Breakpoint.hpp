////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSP_MEASUREMENT_TOOLS_BREAKPOINT_HPP
#define CSP_MEASUREMENT_TOOLS_BREAKPOINT_HPP

#include "Vector2f.hpp"

namespace csp::measurementtools {

struct Arc;
class VoronoiGenerator;

using Edge = std::pair<Vector2f, Vector2f>;

class Breakpoint {
 public:
  Breakpoint();
  Breakpoint(Arc* left, Arc* right, VoronoiGenerator* generator);

  Vector2f const& position() const;
  Edge            finishEdge(Vector2f const& end) const;

  Arc* mLeftArc{nullptr};
  Arc* mRightArc{nullptr};

  Breakpoint* mLeftChild{nullptr};
  Breakpoint* mRightChild{nullptr};
  Breakpoint* mParent{nullptr};

 private:
  void updatePosition() const;

  VoronoiGenerator* mGenerator{nullptr};

  mutable double   mSweepline{-1.0};
  mutable Vector2f mPosition;

  Vector2f mStart;
};
} // namespace csp::measurementtools
#endif // CSP_MEASUREMENT_TOOLS_BREAKPOINT_HPP
