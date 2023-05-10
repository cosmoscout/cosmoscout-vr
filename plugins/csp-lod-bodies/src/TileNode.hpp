////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSP_LOD_BODIES_TILENODE_HPP
#define CSP_LOD_BODIES_TILENODE_HPP

#include "BaseTileData.hpp"

namespace csp::lodbodies {

/// Node in a quad tree of tiles. It stores pointers to its four child nodes (if present), the
/// parent TileNode (unless it is a root node) and to the tile of data associated with this node.
class TileNode {

 public:
  explicit TileNode(TileId const& tileId);

  virtual ~TileNode() = default;

  TileNode(TileNode const& other) = delete;
  TileNode(TileNode&& other)      = default;

  TileNode& operator=(TileNode const& other) = delete;
  TileNode& operator=(TileNode&& other) = default;

  /// Returns the tile data assigned to this. Can be null.
  std::shared_ptr<BaseTileData> const&              getTileData(TileDataType type) const;
  PerDataType<std::shared_ptr<BaseTileData>> const& getTileData() const;

  /// Assigns data to this tile.
  void setTileData(std::shared_ptr<BaseTileData> tile);

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

  /// These are computed based on the TileId given to the constructor and are required by the
  /// TileRender.
  glm::ivec3 const&                getTileOffsetScale() const;
  glm::ivec2 const&                getTileF1F2() const;
  std::array<glm::dvec2, 4> const& getCornersLngLat() const;

  /// Returns if the node is refined, i.e. if its children are loaded.
  bool childrenAvailable() const;

 private:
  TileId                                   mTileId{};
  TileNode*                                mParent{nullptr};
  std::array<std::unique_ptr<TileNode>, 4> mChildren;

  // The actual data for the tile node is stored here. It uses a shared pointer as it is also stored
  // in the upload queue of the TileTextureArray.
  PerDataType<std::shared_ptr<BaseTileData>> mTileData;

  // These are used for visibility checks.
  std::unique_ptr<MinMaxPyramid> mMinMaxPyramid;
  BoundingBox<double>            mTb;
  bool                           mHasBounds{false};

  // These are precomputed at construction time and are required during rendering.
  glm::ivec3                mTileOffsetScale;
  glm::ivec2                mTileF1F2;
  std::array<glm::dvec2, 4> mCornersLngLat;

  int mLastFrame{-1};
};

} // namespace csp::lodbodies

#endif // CSP_LOD_BODIES_TILENODE_HPP
