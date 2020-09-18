////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CSP_MEASUREMENT_TOOLS_BREAKPOINTTREE_HPP
#define CSP_MEASUREMENT_TOOLS_BREAKPOINTTREE_HPP

#include "Vector2f.hpp"

#include <vector>

namespace csp::measurementtools {

using Edge = std::pair<Vector2f, Vector2f>;

class Breakpoint;
struct Arc;

class BreakpointTree {
 public:
  BreakpointTree();

  BreakpointTree(BreakpointTree const& other) = default;
  BreakpointTree(BreakpointTree&& other)      = default;

  BreakpointTree& operator=(BreakpointTree const& other) = default;
  BreakpointTree& operator=(BreakpointTree&& other) = default;

  ~BreakpointTree();

  void insert(Breakpoint* point);
  void remove(Breakpoint* point);
  void finishAll(std::vector<Edge>& edges);

  Arc* getArcAt(double x) const;

  bool empty() const;

 private:
  void        insert(Breakpoint* newNode, Breakpoint* atNode);
  Breakpoint* getNearestNode(double x, Breakpoint* current) const;
  void        finishAll(std::vector<Edge>& edges, Breakpoint* atNode);
  void        clear(Breakpoint* atNode);

  void attachRightOf(Breakpoint* newNode, Breakpoint* atNode);
  void attachLeftOf(Breakpoint* newNode, Breakpoint* atNode);

  Breakpoint* mRoot{nullptr};
};
} // namespace csp::measurementtools
#endif // CSP_MEASUREMENT_TOOLS_BREAKPOINTTREE_HPP
