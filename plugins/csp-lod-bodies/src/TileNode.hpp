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

  /// Returns the tile data owned by this, or NULL if there is no such tile.
  TileBase* getTileData() const;

  /// Sets the tile data to be owned by this. Exclusive ownership of the tile is taken by this and
  /// when this TileNode is destroyed the tile is destroyed as well.
  void setTileData(std::unique_ptr<TileBase> tile);

  /// Returns the child at childIdx (must be in [0, 3]).
  TileNode* getChild(int childIdx) const;

  /// Sets the child at childIdx (must be in [0, 3]). If there is already a child at childIdx it is
  /// destroyed and replaced.
  void setChild(int childIdx, TileNode* child);

  TileNode* getParent() const;

  /// Returns if the node is refined, i.e. if its children are loaded.
  bool isRefined() const;

 private:
  void setParent(TileNode* parent);

  std::unique_ptr<TileBase>                mTileData;
  TileNode*                                mParent{nullptr};
  std::array<std::unique_ptr<TileNode>, 4> mChildren;
};

} // namespace csp::lodbodies

#endif // CSP_LOD_BODIES_TILENODE_HPP
