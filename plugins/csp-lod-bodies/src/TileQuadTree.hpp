////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CSP_LOD_BODIES_TILEQUADTREE_HPP
#define CSP_LOD_BODIES_TILEQUADTREE_HPP

#include "TileNode.hpp"

namespace csp::lodbodies {

/// Stores root nodes of 12 quad trees (i.e. it follows the HEALPix scheme).
class TileQuadTree {
 public:
  static int const sNumRoots = 12;

  /// Returns the TileNode that is the root of the tree idx.
  TileNode* getRoot(int idx) const;

  /// Gives up ownership and returns the TileNode that is the root of the tree idx.
  /// It is now the callers responsibility to correctly dispose of the node.
  TileNode* releaseRoot(int idx);

  /// Sets the TileNode that is the root of tree idx. Exclusive ownership of the node is taken by
  /// this.
  void setRoot(int idx, TileNode* root);

 private:
  std::array<std::unique_ptr<TileNode>, 12> mRoots;
};

/// Inserts node into tree and returns true if it succeeded, false otherwise. Insertion can fail if
/// not all parents of node are currently in tree.
bool insertNode(TileQuadTree* tree, TileNode* node);

/// Removes node from tree and returns true if it succeeded, false otherwise. If true is returned
/// the object pointed to by node does NOT exist any longer. Removal can fail if node does not have
/// a valid parent and is not a root.
bool removeNode(TileQuadTree* tree, TileNode* node);

} // namespace csp::lodbodies

#endif // CSP_LOD_BODIES_TILEQUADTREE_HPP
