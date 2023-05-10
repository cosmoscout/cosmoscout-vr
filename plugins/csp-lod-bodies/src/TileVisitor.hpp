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

/// Base class template for TileQuadTree visitors. It is a base class template that should be
/// instantiated with the deriving class as template argument, i.e.:
///
/// @code{.cpp}
/// class MyVisitor : public TileVisitor<MyVisitor> { };
/// @endcode
///
/// This is the CRTP (curiously recurring template pattern) pattern and allows TileVistor to call
/// member functions of DerivedT without having to resort to runtime polymorphism (i.e. virtual
/// functions).
///
/// The functions preTraverse, postTraverse, preVisitRoot, postVisitRoot, preVisit, and postVisit
/// are called during traversal before/after the traversal as a whole, visiting a root node, and
/// visiting a non-root node respectively. The order of calls looks like:
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
///
/// A common requirement for tree traversal is maintaining a state stack. In order to allow DerivedT
/// to extend the state it must implement three additional functions: pushState, popState, and
/// getState and maintain a stack of objects derived from TileVisitor::StateBase.
///
/// The default implementation of the functions pushState, popState does not actually maintain a
/// stack, but uses a single StateBase object - this is sufficient to provide correct information to
/// the preVisit / preVisitRoot callbacks, but the postVisit / postVisitRoot ones will not see
/// correct values!
///
/// If the "callback" functions (pre/postTraverse, pre/postVisit) are protected or private in
/// DerivedT make TileVisitor a friend class so that it can call these functions.
class TileVisitor {
 public:
  explicit TileVisitor(TileQuadTree* tree);

  /// Start traversal of the trees passed to the constructor.
  void visit();

 protected:
  void visitRoot(TileNode* root);
  void visitLevel(TileNode* node);

  /// Called before visiting the first root node. Returns if traversal should commence or
  /// not.
  /// Reimplement in the derived class, the default just returns true.
  virtual bool preTraverse();

  /// Called after visiting the last node. Reimplement in the derived class, the default
  /// just does
  /// nothing.
  virtual void postTraverse();

  /// Called for each root node visited, before visiting any children. Returns if any
  /// children
  /// should be visited (true) or skipped (false).
  /// Reimplement in the derived class, the default just returns false.
  ///
  /// For finer grained control over which children to visit, set the corresponding entries of
  /// StateBase::children to true (visit child - the default) or false (skip child). These
  /// entries are only considered if this functions returns true.
  virtual bool preVisitRoot(TileNode* root);

  /// Called for each root node visited, after visiting any children. Reimplement in the
  /// derived
  /// class, the default just does nothing.
  virtual void postVisitRoot(TileNode* root);

  /// Called for each non-root node visited, before visiting the child nodes.
  /// Returns if children should be visited (true) or skipped (false).
  /// Reimplement in the derived class, the default just returns false.
  ///
  /// For finer grained control over which children to visit, set the corresponding entries
  /// of StateBase::children to true (visit child - the default) or false (skip child).
  /// These entries are only considered if this functions returns true.
  virtual bool preVisit(TileNode* node);

  /// Called for each node visited, after visiting the child nodes. Reimplement in the
  /// derived
  /// class, the default just does nothing.
  virtual void postVisit(TileNode* node);

  TileQuadTree* mTree;
};

} // namespace csp::lodbodies

#endif // CSP_LOD_BODIES_TILEVISITOR_HPP
