////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSP_LOD_BODIES_TILENODE_HPP
#define CSP_LOD_BODIES_TILENODE_HPP

#include "TileBase.hpp"

namespace csp::lodbodies {

/// Node in a quad tree of tiles. It stores pointers to its four child nodes (if present), the
/// parent TileNode (unless it is a root node) and to the tile of data associated with this node.
class TileNode {

 public:
  explicit TileNode();
  explicit TileNode(TileBase* tile);
  explicit TileNode(std::unique_ptr<TileBase>&& tile);

  virtual ~TileNode() = default;

  TileNode(TileNode const& other) = delete;
  TileNode(TileNode&& other)      = default;

  TileNode& operator=(TileNode const& other) = delete;
  TileNode& operator=(TileNode&& other) = default;

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

  TileNode* getParent() const;

 private:
  void setParent(TileNode* parent);

  std::unique_ptr<TileBase>                mTile;
  TileNode*                                mParent{nullptr};
  std::array<std::unique_ptr<TileNode>, 4> mChildren;
};

/// Returns if the @a node is refined, i.e. if its children are loaded.
bool isRefined(TileNode const& node);

} // namespace csp::lodbodies

#endif // CSP_LOD_BODIES_TILENODE_HPP
