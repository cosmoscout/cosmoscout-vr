////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "TileQuadTree.hpp"

#include "HEALPix.hpp"

namespace csp::lodbodies {

////////////////////////////////////////////////////////////////////////////////////////////////////

TileNode* TileQuadTree::getRoot(int idx) const {
  return mRoots.at(idx).get();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void TileQuadTree::setRoot(int idx, TileNode* root) {
  mRoots.at(idx).reset(root);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool insertNode(TileQuadTree* tree, TileNode* node) {
  bool          result = true;
  TileId const& tileId = node->getTileId();

  if (tileId.level() == 0) {
    assert(tree->getRoot(HEALPix::getRootIdx(tileId)) == nullptr);

    tree->setRoot(HEALPix::getRootIdx(tileId), node);
  } else {
    TileNode* parent = tree->getRoot(HEALPix::getRootIdx(tileId));

    for (int i = 1; i < tileId.level() && parent; ++i) {
      parent = parent->getChild(HEALPix::getChildIdxAtLevel(tileId, i));
    }

    if (parent) {
      // Catch cases where an existing child would be overwritten
      assert(parent->getChild(HEALPix::getChildIdxAtLevel(tileId, tileId.level())) == nullptr);

      parent->setChild(HEALPix::getChildIdxAtLevel(tileId, tileId.level()), node);
    } else {
      result = false;
    }
  }

  return result;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool removeNode(TileQuadTree* tree, TileNode* node) {
  bool      result = false;
  TileNode* parent = node->getParent();

  // node must either have a parent or be a root of the tree (otherwise node
  // is not in tree or the data structure is corrupt).
  if (parent) {
    int childIdx = HEALPix::getChildIdx(node->getTileId());
    assert(parent->getChild(childIdx) == node);

    parent->setChild(childIdx, nullptr);
    result = true;
  } else {
    int childIdx = HEALPix::getChildIdx(node->getTileId());
    assert(tree->getRoot(childIdx) == node);

    tree->setRoot(childIdx, nullptr);
    result = true;
  }

  return result;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::lodbodies
