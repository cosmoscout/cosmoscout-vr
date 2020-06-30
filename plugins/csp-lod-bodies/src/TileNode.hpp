////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CSP_LOD_BODIES_TILENODE_HPP
#define CSP_LOD_BODIES_TILENODE_HPP

#include "TileBase.hpp"

#include <boost/move/utility.hpp>

namespace csp::lodbodies {

/// Node in a quad tree of tiles. It stores pointers to its four child nodes (if present), the
/// parent TileNode (unless it is a root node) and to the tile of data associated with this node.
class TileNode {
 private:
  BOOST_MOVABLE_BUT_NOT_COPYABLE(TileNode)

 public:
  explicit TileNode();
  explicit TileNode(TileBase* tile, int childMaxLevel = -1);
  explicit TileNode(std::unique_ptr<TileBase>&& tile, int childMaxLevel = -1);

  // move constructor -- disabled: triggers a bug with gcc 4.3?
  //     TileNode(BOOST_RV_REF(TileNode) source);

  // move assignment -- disabled: triggers a bug with gcc 4.3?
  //     TileNode& operator=(BOOST_RV_REF(TileNode) source);

  int           getLevel() const;
  glm::int64    getPatchIdx() const;
  TileId const& getTileId() const;

  std::type_info const& getTileTypeId() const;
  TileDataType          getTileDataType() const;

  /// Returns the tile owned by this, or NULL if there is no such tile.
  TileBase* getTile() const;

  /// Gives up ownership of the tile owned by this and returns it, or NULL if there is no such tile.
  /// It is now the callers responsibility to correctly dispose of the tile.
  TileBase* releaseTile();

  /// Sets the tile to be owned by this. Exclusive ownership of the tile is taken by this and when
  /// this TileNode is destroyed the tile is destroyed as well.
  void setTile(std::unique_ptr<TileBase> tile);

  /// Returns the child at childIdx (must be in [0, 3]).
  TileNode* getChild(int childIdx) const;

  /// Gives up ownership of the child at childIdx (must be in [0, 3]) and returns it. It is now the
  /// callers responsibility to correctly dispose of the child node.
  TileNode* releaseChild(int childIdx);

  /// Sets the child at childIdx (must be in [0, 3]). If there is already a child at childIdx it is
  /// destroyed and replaced.
  void setChild(int childIdx, TileNode* child);

  /// Returns largest level of any child.
  int getChildMaxLevel() const;

  /// Sets largest level of any child.
  void setChildMaxLevel(int maxLevel);

  TileNode* getParent() const;

 private:
  void setParent(TileNode* parent);

  std::unique_ptr<TileBase>                mTile;
  TileNode*                                mParent;
  std::array<std::unique_ptr<TileNode>, 4> mChildren;
  int                                      mChildMaxLevel;
};

/// Returns if the node is a leaf, i.e. if it can not be further refined.
///
/// This is the case when @code{.cpp} node.getLevel() == node.getChildMaxLevel() @endcode
bool isLeaf(TileNode const& node);

/// Returns if the node is an inner node, i.e. not a leaf node.
///
/// This is the case when @code{.cpp} node.getLevel() < node.getChildMaxLevel() @endcode
bool isInner(TileNode const& node);

/// Returns if the @a node is refined, i.e. if its children are loaded.
bool isRefined(TileNode const& node);

} // namespace csp::lodbodies

#endif // CSP_LOD_BODIES_TILENODE_HPP
