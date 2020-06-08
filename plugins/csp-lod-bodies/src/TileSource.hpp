////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CSP_LOD_BODIES_TILESOURCE_HPP
#define CSP_LOD_BODIES_TILESOURCE_HPP

#include "TileDataType.hpp"
#include "TileId.hpp"

namespace csp::lodbodies {

class TileNode;

/// Base class/interface for sources of tile data. Defines interfaces for synchronous (blocking) and
/// asynchronous (non-blocking) loading of tiles, optionally allocating objects as needed or reusing
/// pre-existing ones.
class TileSource {
 public:
  /// Type of the callback functor that can be passed to loadTileAsync.
  using OnLoadCallback = std::function<void(TileSource*, int, glm::int64, TileNode*)>;

  TileSource() = default;

  TileSource(TileSource const& other)     = default;
  TileSource(TileSource&& other) noexcept = default;

  TileSource& operator=(TileSource const& other) = default;
  TileSource& operator=(TileSource&& other) noexcept = default;

  virtual ~TileSource() = default;

  /// Perform initialization of the source. Must be called before the first tile is requested with
  /// loadTile or loadTileAsync. Sub-classes need to make sure that it is safe to call this
  /// repeatedly.
  virtual void init() = 0;

  /// Performs shutdown of the source. No further calls to loadTile or loadTileAsync are allowed
  /// without calling init first. Sub-classes need to make sure that it is safe to call this
  /// repeatedly.
  virtual void fini() = 0;

  /// Returns the enum value for the data type stored in tiles produced by this TileSource.
  virtual TileDataType getDataType() const = 0;

  /// Loads a node with given level and patchIx synchronously (i.e. the call blocks until data is
  /// loaded). Optionally the node to store data in is passed as node - it must own a Tile of
  /// correct type or not own a tile at all (in which case a new one is allocated). If node is
  /// a nullptr a new node is allocated.
  virtual TileNode* loadTile(int level, glm::int64 patchIdx) = 0;

  /// Loads a node with given level and patchIdx asynchronously (i.e. the call returns immediately).
  /// Optionally the node to store data in is passed as node - it must own a Tile of correct type or
  /// not own a tile at all (in which case a new one is allocated). If node is a nullptr a new node
  /// is allocated. Once the node is loaded the given OnLoadCallack is invoked.
  virtual void loadTileAsync(int level, glm::int64 patchIdx, OnLoadCallback cb) = 0;

  /// Returns the number of currently active async requests.
  virtual int getPendingRequests() = 0;

  /// Derived classes should check whether the given TileSource has the same type and members. This
  /// is used to prevent redundant tile source reloading.
  virtual bool isSame(TileSource const* other) const = 0;
};

} // namespace csp::lodbodies

#endif // CSP_LOD_BODIES_TILESOURCE_HPP
