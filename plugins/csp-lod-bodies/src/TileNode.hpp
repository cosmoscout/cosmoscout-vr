////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSP_LOD_BODIES_TILENODE_HPP
#define CSP_LOD_BODIES_TILENODE_HPP

#include "TileDataBase.hpp"

namespace csp::lodbodies {

/// Node in a quad tree of tiles. It stores pointers to its four child nodes (if present), the
/// parent TileNode (unless it is a root node) and to the tile of data associated with this node.
class TileNode {

 public:
  explicit TileNode() = default;
  explicit TileNode(TileId const& tileId);

  virtual ~TileNode() = default;

  TileNode(TileNode const& other) = delete;
  TileNode(TileNode&& other)      = default;

  TileNode& operator=(TileNode const& other) = delete;
  TileNode& operator=(TileNode&& other) = default;

  /// Returns the tile data owned by this, or NULL if there is no such tile.
  TileDataBase*                                     getTileData(TileDataType type) const;
  PerDataType<std::unique_ptr<TileDataBase>> const& getTileData() const;

  /// Sets the tile data to be owned by this. Exclusive ownership of the tile is taken by this and
  /// when this TileNode is destroyed the tile is destroyed as well.
  void setTileData(std::unique_ptr<TileDataBase> tile);

  /// Returns the child at childIdx (must be in [0, 3]).
  TileNode* getChild(int childIdx) const;

  /// Sets the child at childIdx (must be in [0, 3]). If there is already a child at childIdx it is
  /// destroyed and replaced.
  void setChild(int childIdx, TileNode* child);

  TileNode* getParent() const;

  int           getLevel() const;
  glm::int64    getPatchIdx() const;
  TileId const& getTileId() const;

  int  getLastFrame() const;
  void setLastFrame(int frame);
  int  getAge(int frame) const;

  BoundingBox<double> const& getBounds() const;
  void                       setBounds(BoundingBox<double> const& tb);
  void                       removeBounds();
  bool                       hasBounds() const;

  MinMaxPyramid* getMinMaxPyramid() const;
  void           setMinMaxPyramid(std::unique_ptr<MinMaxPyramid> pyramid);

  /// Returns if the node is refined, i.e. if its children are loaded.
  bool isRefined() const;

 private:
  void                                     setParent(TileNode* parent);
  TileId                                   mTileId{};
  TileNode*                                mParent{nullptr};
  std::array<std::unique_ptr<TileNode>, 4> mChildren;

  PerDataType<std::unique_ptr<TileDataBase>> mTileData;

  std::unique_ptr<MinMaxPyramid> mMinMaxPyramid;
  BoundingBox<double>            mTb;
  bool                           mHasBounds{false};
  int                            mLastFrame{-1};
};

} // namespace csp::lodbodies

#endif // CSP_LOD_BODIES_TILENODE_HPP
