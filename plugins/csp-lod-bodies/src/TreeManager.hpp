////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSP_LOD_BODIES_TREEMANAGER_HPP
#define CSP_LOD_BODIES_TREEMANAGER_HPP

#include "TileId.hpp"
#include "TileQuadTree.hpp"

#include <mutex>
#include <string>
#include <unordered_set>
#include <vector>

namespace csp::lodbodies {

class TileNode;
class TileSource;
class BaseTileData;
class GLResources;
class TileTextureArray;

/// Manages a TileQuadTree of TileNodes requested from a TileSource.
///
/// Tiles to load from the configured TileSource are passed in with a call to request and previously
/// (asynchronously) loaded tiles are merged into the TileQuadTree with a call to update.
///
/// In addition to managing the loading of tiles and inserting them into the managed TileQuadTree
/// this also keeps track of the "age" of nodes. A nodes age is measured in frames since the last
/// time it was used - other classes mark nodes as used (e.g. LODVisitor when testing visibility of
/// a node).
///
/// In order to quickly find "old" nodes a vector of node pointers is used. The vector is sorted so
/// that the oldest nodes are at the back and those are removed if their age exceeds a certain
/// threshold (see TreeManager::prune).
class TreeManager {
 public:
  explicit TreeManager(std::shared_ptr<GLResources> glResources);

  TreeManager(TreeManager const& other) = delete;
  TreeManager(TreeManager&& other)      = delete;

  TreeManager& operator=(TreeManager const& other) = delete;
  TreeManager& operator=(TreeManager&& other) = delete;

  virtual ~TreeManager() = default;

  /// Set tile source to use.
  void setSource(TileDataType type, TileSource* src);

  /// Returns pointer to the TileQuadTree managed by this.
  TileQuadTree* getTree();

  std::shared_ptr<GLResources> const& getGLResources() const;

  /// Request data tiles with indices tileIds to be loaded and queued to be merged into the quad
  /// tree (with a subsequent call to update).
  void request(std::vector<TileId> const& tileIds);

  /// Update the TileQuadTree managed by this with the tiles that have been loaded from the
  /// TileSource since the last call to update.
  void update();

  /// Removes all nodes from the tree and frees data associated with them.
  void clear();

  void setFrameCount(int frameCount);

 private:
  struct AgeLess;

  /// Tracks a node and the frame it was loaded in - for nodes that can not immediately be merged.
  struct NodeAge {
    explicit NodeAge(TileNode* node, int frame);

    TileNode* mNode;
    int       mFrame;
  };

  /// Used as a callback for the TileSource to call when a node is loaded.
  void onDataLoaded(TileId const& tileId, std::shared_ptr<BaseTileData> tileData);

  /// Helper function to handle processing after node is successfully inserted into the managed
  /// TileQuadTree.
  void onNodeInserted(TileNode* node);

  /// Helper function to free resources associated with node.
  void releaseResources(TileNode* node);

  /// Remove nodes from the managed TileQuadTree that have not been used for a number of frames.
  /// Sort tiles by age (frames since last use, see TreeManager::AgeLess for details) and
  /// removes those considered too "old".
  void prune();

  /// Merge nodes loaded since the last merge into the managed TileQuadTree. It is possible that a
  /// loaded node can not be inserted into the tree, for example because its parent has been removed
  /// in the meantime. These "unmerged" nodes are kept around in mUnmergedNodes for a few frames, in
  /// case the parent node is loaded in the meantime. If this "grace period" has expired and the
  /// node still cannot be inserted into the tree it is deleted.
  void merge();

  std::shared_ptr<GLResources> mGLResources;
  std::vector<TileNode*>       mNodes;

  TileQuadTree             mTree;
  PerDataType<TileSource*> mTileDataSources;

  std::unordered_map<TileId, TileNode*> mPendingTiles;
  std::vector<NodeAge>                  mUnmergedNodes;
  std::vector<TileNode*>                mLoadedNodes;

  std::mutex mSourcesMtx;
  std::mutex mLoadedMtx;
  std::mutex mPendingMtx;

  int  mFrameCount;
  bool mAsyncLoading;
};

} // namespace csp::lodbodies

#endif // CSP_LOD_BODIES_TREEMANAGER_HPP
