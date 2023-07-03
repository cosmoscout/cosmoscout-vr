////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSP_LOD_BODIES_TILEVISITOR_HPP
#define CSP_LOD_BODIES_TILEVISITOR_HPP

namespace csp::lodbodies {

class TileNode;
class TileQuadTree;

/// Base class for TileQuadTree visitors. The functions preTraverse, postTraverse, preVisitRoot,
/// postVisitRoot, preVisit, and postVisit are called during traversal before/after the traversal as
/// a whole, visiting a root node, and visiting a non-root node respectively. The order of calls
/// looks like:
///
/// @code{.cpp}
/// preTraverse()
///   preVisitRoot()        // root 0
///     preVisit()          // root 0 - child 0
///     postVisit()         // root 0 - child 0
///     preVisit()          // root 0 - child 1
///     postVisit()         // root 0 - child 1
///     // ...
///   postVisitRoot()       // root 0
/// postTraverse()
/// @endcode
class TileVisitor {
 public:
  explicit TileVisitor(TileQuadTree* tree);

  /// Start traversal of the trees passed to the constructor.
  void visit();

 protected:
  void visitRoot(TileNode* root);
  void visitLevel(TileNode* node);

  /// Called before visiting the first root node. Returns if traversal should commence or not.
  /// Reimplement in the derived class, the default just returns true.
  virtual bool preTraverse();

  /// Called after visiting the last node. Reimplement in the derived class, the default just does
  /// nothing.
  virtual void postTraverse();

  /// Called for each root node visited, before visiting any children. Returns if any children
  /// should be visited (true) or skipped (false). Reimplement in the derived class, the default
  /// just returns false.
  virtual bool preVisitRoot(TileNode* root);

  /// Called for each root node visited, after visiting any children. Reimplement in the derived
  /// class, the default just does nothing.
  virtual void postVisitRoot(TileNode* root);

  /// Called for each non-root node visited, before visiting the child nodes. Returns if children
  /// should be visited (true) or skipped (false). Reimplement in the derived class, the default
  /// just returns false.
  virtual bool preVisit(TileNode* node);

  /// Called for each node visited, after visiting the child nodes. Reimplement in the derived
  /// class, the default just does nothing.
  virtual void postVisit(TileNode* node);

  TileQuadTree* mTree;
};

} // namespace csp::lodbodies

#endif // CSP_LOD_BODIES_TILEVISITOR_HPP
