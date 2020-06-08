////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CSP_LOD_BODIES_TILEVISITOR_HPP
#define CSP_LOD_BODIES_TILEVISITOR_HPP

#include "HEALPix.hpp"
#include "TileId.hpp"
#include "TileQuadTree.hpp"

#include <boost/assert.hpp>

namespace csp::lodbodies {

class TileNode;

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
template <typename DerivedT>
class TileVisitor {
 public:
  using DerivedType = DerivedT;

  explicit TileVisitor(TileQuadTree* treeDEM, TileQuadTree* treeIMG = nullptr);

  /// Start traversal of the trees passed to the constructor.
  void visit();

  void          setTreeDEM(TileQuadTree* tree);
  TileQuadTree* getTreeDEM() const;

  void          setTreeIMG(TileQuadTree* tree);
  TileQuadTree* getTreeIMG() const;

  TileNode* getNodeDEM() const;
  TileNode* getNodeIMG() const;

  TileId const& getTileId() const;
  int           getLevel() const;
  glm::int64    getPatchIdx() const;

 protected:
  class StateBase {
   public:
    std::array<bool, 4> mChildren{};
    TileNode*           mNodeDEM{};
    TileNode*           mNodeIMG{};
    TileId              mTileId;
  };

  /// Convenience access to members of DerivedType.
  DerivedType&       self();
  DerivedType const& self() const;

  void visitRoot(TileNode* rootDEM, TileNode* rootIMG, TileId tileId);
  void visitLevel(TileNode* nodeDEM, TileNode* nodeIMG, TileId tileId);

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
  virtual bool preVisitRoot(TileId const& tileId);

  /// Called for each root node visited, after visiting any children. Reimplement in the
  /// derived
  /// class, the default just does nothing.
  virtual void postVisitRoot(TileId const& tileId);

  /// Called for each non-root node visited, before visiting the child nodes.
  /// Returns if children should be visited (true) or skipped (false).
  /// Reimplement in the derived class, the default just returns false.
  ///
  /// For finer grained control over which children to visit, set the corresponding entries
  /// of StateBase::children to true (visit child - the default) or false (skip child).
  /// These entries are only considered if this functions returns true.
  virtual bool preVisit(TileId const& tileId);

  /// Called for each node visited, after visiting the child nodes. Reimplement in the
  /// derived
  /// class, the default just does nothing.
  virtual void postVisit(TileId const& tileId);

  virtual void             pushState();
  virtual void             popState();
  virtual StateBase&       getState();
  virtual StateBase const& getState() const;

  TileQuadTree* mTreeDEM;
  TileQuadTree* mTreeIMG;
  StateBase     mDummyState;
};

template <typename DerivedT>
TileVisitor<DerivedT>::TileVisitor(TileQuadTree* treeDEM, TileQuadTree* treeIMG)
    : mTreeDEM(treeDEM)
    , mTreeIMG(treeIMG)
    , mDummyState() {
}

template <typename DerivedT>
void TileVisitor<DerivedT>::visit() {
  if (self().preTraverse()) {
    for (int i = 0; i < TileQuadTree::sNumRoots; ++i) {
      TileNode* rootDEM = mTreeDEM->getRoot(i);
      TileNode* rootIMG = mTreeIMG ? mTreeIMG->getRoot(i) : nullptr;

      // if (i==7)
      visitRoot(rootDEM, rootIMG, TileId(0, i));
    }
  }

  self().postTraverse();
}

template <typename DerivedT>
void TileVisitor<DerivedT>::setTreeDEM(TileQuadTree* tree) {
  mTreeDEM = tree;
}

template <typename DerivedT>
TileQuadTree* TileVisitor<DerivedT>::getTreeDEM() const {
  return mTreeDEM;
}

template <typename DerivedT>
void TileVisitor<DerivedT>::setTreeIMG(TileQuadTree* tree) {
  mTreeIMG = tree;
}

template <typename DerivedT>
TileQuadTree* TileVisitor<DerivedT>::getTreeIMG() const {
  return mTreeIMG;
}

template <typename DerivedT>
TileNode* TileVisitor<DerivedT>::getNodeDEM() const {
  return self().getState().mNodeDEM;
}

template <typename DerivedT>
TileNode* TileVisitor<DerivedT>::getNodeIMG() const {
  return self().getState().mNodeIMG;
}

template <typename DerivedT>
TileId const& TileVisitor<DerivedT>::getTileId() const {
  return self().getState().mTileId;
}

template <typename DerivedT>
int TileVisitor<DerivedT>::getLevel() const {
  return getTileId().level();
}

template <typename DerivedT>
glm::int64 TileVisitor<DerivedT>::getPatchIdx() const {
  return getTileId().patchIdx();
}

template <typename DerivedT>
typename TileVisitor<DerivedT>::DerivedType& TileVisitor<DerivedT>::self() {
  return *static_cast<DerivedType*>(this);
}

template <typename DerivedT>
typename TileVisitor<DerivedT>::DerivedType const& TileVisitor<DerivedT>::self() const {
  return *static_cast<DerivedType const*>(this);
}

template <typename DerivedT>
void TileVisitor<DerivedT>::visitRoot(TileNode* rootDEM, TileNode* rootIMG, TileId tileId) {
  // check that nodes have expected level - if this triggers the trees are
  // corrupted
  assert(rootDEM == NULL || rootDEM->getLevel() == tileId.level());
  assert(rootIMG == NULL || rootIMG->getLevel() == tileId.level());

  // push & init state
  self().pushState();
  StateBase& state   = self().getState();
  state.mChildren[0] = true;
  state.mChildren[1] = true;
  state.mChildren[2] = true;
  state.mChildren[3] = true;
  state.mNodeDEM     = rootDEM;
  state.mNodeIMG     = rootIMG;
  state.mTileId      = tileId;

  if (self().preVisitRoot(tileId)) {
    for (int i = 0; i < 4; ++i) {
      if (!state.mChildren[i]) {
        continue;
      }

      TileNode* childDEM = rootDEM ? rootDEM->getChild(i) : nullptr;
      TileNode* childIMG = rootIMG ? rootIMG->getChild(i) : nullptr;

      if (childDEM || childIMG) {
        TileId childId = HEALPix::getChildTileId(tileId, i);
        visitLevel(childDEM, childIMG, childId);
      }
    }
  }

  self().postVisitRoot(tileId);
  self().popState();
}

template <typename DerivedT>
void TileVisitor<DerivedT>::visitLevel(TileNode* nodeDEM, TileNode* nodeIMG, TileId tileId) {
  // check that nodes have expected level - if this triggers the trees are
  // corrupted
  assert(nodeDEM == NULL || nodeDEM->getLevel() == tileId.level());
  assert(nodeIMG == NULL || nodeIMG->getLevel() == tileId.level());

  // push & init state
  self().pushState();
  StateBase& state   = self().getState();
  state.mChildren[0] = true;
  state.mChildren[1] = true;
  state.mChildren[2] = true;
  state.mChildren[3] = true;
  state.mNodeDEM     = nodeDEM;
  state.mNodeIMG     = nodeIMG;
  state.mTileId      = tileId;

  if (self().preVisit(tileId)) {
    for (int i = 0; i < 4; ++i) {
      if (!state.mChildren[i]) {
        continue;
      }

      TileNode* childDEM = nodeDEM ? nodeDEM->getChild(i) : nullptr;
      TileNode* childIMG = nodeIMG ? nodeIMG->getChild(i) : nullptr;

      if (childDEM || childIMG) {
        TileId childId = HEALPix::getChildTileId(tileId, i);
        visitLevel(childDEM, childIMG, childId);
      }
    }
  }

  self().postVisit(tileId);
  self().popState();
}

template <typename DerivedT>
bool TileVisitor<DerivedT>::preTraverse() {
  // default impl - start traversal
  return true;
}

template <typename DerivedT>
void TileVisitor<DerivedT>::postTraverse() {
  // default impl - empty
}

template <typename DerivedT>
bool TileVisitor<DerivedT>::preVisitRoot(TileId const& /*tileId*/) {
  // default impl - do not visit children
  return false;
}

template <typename DerivedT>
void TileVisitor<DerivedT>::postVisitRoot(TileId const& tileId) {
  // default impl - empty
}

template <typename DerivedT>
bool TileVisitor<DerivedT>::preVisit(TileId const& /*tileId*/) {
  // default impl - do not visit children
  return false;
}

template <typename DerivedT>
void TileVisitor<DerivedT>::postVisit(TileId const& tileId) {
  // default impl - empty
}

// ===========================================================================
// state callbacks

template <typename DerivedT>
void TileVisitor<DerivedT>::pushState() {
  // default impl - empty
}

template <typename DerivedT>
void TileVisitor<DerivedT>::popState() {
  // default impl - empty
}

template <typename DerivedT>
typename TileVisitor<DerivedT>::StateBase& TileVisitor<DerivedT>::getState() {
  return mDummyState;
}

template <typename DerivedT>
typename TileVisitor<DerivedT>::StateBase const& TileVisitor<DerivedT>::getState() const {
  return mDummyState;
}

} // namespace csp::lodbodies

#endif // CSP_LOD_BODIES_TILEVISITOR_HPP
