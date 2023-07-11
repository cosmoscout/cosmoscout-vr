////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "TreeManager.hpp"

#include "PlanetParameters.hpp"
#include "TileData.hpp"
#include "TileSource.hpp"
#include "TileTextureArray.hpp"

#include <VistaBase/VistaStreamUtils.h>

#include <algorithm>
#include <utility>

namespace csp::lodbodies {

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace {

////////////////////////////////////////////////////////////////////////////////////////////////////

// The following constants could be made member variables of TreeManager
// in order to allow them to be modified at runtime (possibly with the
// values below as defaults).

// number of frames an unused node is kept in the tree before being removed
int const maxNodeAge = 10;

// number of frames a node that can not directly be merged into the tree
// is kept around
int const maxUnmergedAge = 500;

// number of nodes to pre-allocate data structures
std::size_t const preAllocNodeCount = 500;

// number of nodes to pre-allocate IO data structures
std::size_t const preAllocIONodeCount = 200;

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace

////////////////////////////////////////////////////////////////////////////////////////////////////

// Function object to compare RDMapValue objects by their age.
struct TreeManager::AgeLess {
  explicit AgeLess(int frame);

  bool operator()(TileNode const* lhs, TileNode const* rhs) const;

  int mFrame;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

/* explicit */
TreeManager::AgeLess::AgeLess(int frame)
    : mFrame(frame) {
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool TreeManager::AgeLess::operator()(TileNode const* lhs, TileNode const* rhs) const {
  // sort by age (frames since last use), in case of a tie use
  // tile level - this ensure that child nodes are always sorted
  // after parent nodes

  int ageLHS = lhs->getAge(mFrame);
  int ageRHS = rhs->getAge(mFrame);

  if (ageLHS == ageRHS) {
    return lhs->getLevel() < rhs->getLevel();
  }

  return ageLHS < ageRHS;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

/* explicit */
TreeManager::NodeAge::NodeAge(TileNode* node, int frame)
    : mNode(node)
    , mFrame(frame) {
}

////////////////////////////////////////////////////////////////////////////////////////////////////

/* explicit */
TreeManager::TreeManager(std::shared_ptr<GLResources> glResources)
    : mGLResources(std::move(glResources))
    , mFrameCount(0)
    , mAsyncLoading(true) {

  mNodes.reserve(preAllocNodeCount);
  mUnmergedNodes.reserve(preAllocIONodeCount);
  mLoadedNodes.reserve(preAllocIONodeCount);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void TreeManager::setSource(TileDataType type, TileSource* src) {
  // remove all existing nodes
  clear();

  std::unique_lock<std::mutex> lck(mSourcesMtx);
  mTileDataSources.set(type, src);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void TreeManager::request(std::vector<TileId> const& tileIds) {
  // for each requested tile, check if it is already in the mPendingTiles
  // set (those are tiles that have already been requesed from the tile
  // source), otherwise put the tile in mPendingTiles and ask the source to
  // load the tile.
  // In the case of async loading, register @c onDataLoaded as the callback
  // that the source invokes when the tile is ready.
  std::unique_lock<std::mutex> lck(mPendingMtx);

  for (auto const& tileId : tileIds) {
    if (mPendingTiles.count(tileId) == 0) {
      mPendingTiles[tileId] = new TileNode(tileId);

      for (auto const& src : mTileDataSources.mChannels) {
        if (src) {
          if (mAsyncLoading) {
            src->loadTileAsync(
                tileId, [this](auto id, auto data) { onDataLoaded(id, std::move(data)); });
          } else {
            auto tileData = src->loadTile(tileId);
            onDataLoaded(tileId, std::move(tileData));
          }
        }
      }
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void TreeManager::update() {
  // remove unused nodes - do this before the merge to free up resources
  // that can then be consumed by newly loaded ones.
  prune();

  // insert new nodes
  merge();

  // upload tiles to GPU
  for (auto const& textureArray : mGLResources->mChannels) {
    textureArray->processQueue(5);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void TreeManager::clear() {
  {
    std::unique_lock<std::mutex> lck(mLoadedMtx);
    mLoadedNodes.clear();
  }

  {
    std::unique_lock<std::mutex> lck(mPendingMtx);
    mPendingTiles.clear();
  }

  for (auto* node : mNodes) {
    releaseResources(node);
  }

  mNodes.clear();

  for (int i = 0; i < TileQuadTree::sNumRoots; ++i) {
    mTree.setRoot(i, nullptr);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void TreeManager::onDataLoaded(TileId const& tileId, std::shared_ptr<BaseTileData> tileData) {

  // If tile loading failed, discard the data.
  if (!tileData) {
    std::unique_lock<std::mutex> lck(mPendingMtx);
    mPendingTiles.erase(tileId);
    return;
  }

  TileNode* node{};

  {
    std::unique_lock<std::mutex> lck(mPendingMtx);

    // If the data is not need the tile anymore, discard it.
    auto it = mPendingTiles.find(tileId);
    if (it == mPendingTiles.end()) {
      return;
    }

    node = it->second;
  }

  if (tileData->getDataType() == TileDataType::eElevation) {
    auto demdata = dynamic_cast<TileData<float>*>(tileData.get());
    node->setMinMaxPyramid(std::make_unique<MinMaxPyramid>(demdata));
  }

  node->setTileData(std::move(tileData));

  auto dem = node->getTileData(TileDataType::eElevation);
  auto img = node->getTileData(TileDataType::eColor);

  bool hasColorChannel = false;

  {
    std::unique_lock<std::mutex> lck(mSourcesMtx);
    hasColorChannel = mTileDataSources.get(TileDataType::eColor);
  }

  if (dem && (img || !hasColorChannel)) {
    // Only add node to list of loaded nodes, actual insertion into the
    // quad-tree is done in merge().
    // This ensures that the tree is not modified at unpredictable moments
    // in time (for example while a traversal is in progress).
    std::unique_lock<std::mutex> lck(mLoadedMtx);
    mLoadedNodes.push_back(node);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void TreeManager::onNodeInserted(TileNode* node) {
  if (node->getParent()) {
    TileNode* parent = node->getParent();
    assert(parent != nullptr);

    node->setLastFrame(parent->getLastFrame());
  }

  mNodes.push_back(node);

  for (auto const& res : mGLResources->mChannels) {
    auto data = node->getTileData(res->getDataType());
    if (data) {
      res->allocateGPU(data);
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void TreeManager::releaseResources(TileNode* node) {
  for (auto const& res : mGLResources->mChannels) {
    auto data = node->getTileData(res->getDataType());
    if (data) {
      res->releaseGPU(data);
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void TreeManager::prune() {
  // sort by age, oldest nodes at the back
  std::sort(mNodes.begin(), mNodes.end(), AgeLess(mFrameCount));
  int count = 0;

  while (!mNodes.empty()) {
    TileNode* node = mNodes.back();

    // remove unnused nodes, but never root nodes
    if (node->getAge(mFrameCount) > maxNodeAge && node->getLevel() > 0) {
      releaseResources(node);

      if (!removeNode(&mTree, node)) {
        vstr::errp() << "[TreeManager::prune] Failed to remove node " << node << "!" << std::endl;
      }

      // remove entries for node from internal data structures
      mNodes.pop_back();
      ++count;
    } else {
      // The node at the back of mAgeStore can not be removed - stop
      // The sorting of mAgeStore ensures that there is no node that
      // could be removed beyond this point.
      break;
    }
  }

  if (count > 0) {
#if !defined(NDEBUG) && !defined(VISTAPLANET_NO_VERBOSE)
    vstr::outi() << "[TreeManager::prune] nodes removed/kept " << count << " / " << mRdMap.size()
                 << std::endl;
#endif
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void TreeManager::merge() {
  // exchange mLoadedNodes and mergeNodes so the lock need not be held
  // for the duration of the whole merge
  std::vector<TileNode*> mergeNodes;
  {
    std::unique_lock<std::mutex> lck(mLoadedMtx);

    mergeNodes = mLoadedNodes;
    mLoadedNodes.clear();
  }

  int merged   = 0;
  int unmerged = 0;

  for (auto& node : mergeNodes) {
    assert(node != nullptr);
    assert(node->getTile() != nullptr);

    if (insertNode(&mTree, node)) {
      onNodeInserted(node);
      ++merged;

      std::unique_lock<std::mutex> lck(mPendingMtx);
      mPendingTiles.erase(node->getTileId());

      node = nullptr;
    } else {
      // keep track of nodes that could not be inserted, e.g. because
      // their parent is currently not loaded
      ++unmerged;
    }
  }

  // attempt to merge unmerged nodes from previous frames
  // Go through list of unmerged nodes and attempt to insert them into the
  // tree. If that fails and the nodes age exceeds maxUnmergedAge, it is
  // discarded.
  for (std::size_t i = 0; i < mUnmergedNodes.size();) {
    TileNode* node = mUnmergedNodes[i].mNode;

    if (insertNode(&mTree, node)) {
      // insert succeeded, remove from pending and unmerged and
      // associate render data with node
      mPendingTiles.erase(node->getTileId());
      mUnmergedNodes.erase(mUnmergedNodes.begin() + i);

      onNodeInserted(node);
    } else if ((mFrameCount - mUnmergedNodes[i].mFrame) > maxUnmergedAge) {
      // node is waiting for too long to be merged - discard it
      mPendingTiles.erase(node->getTileId());
      mUnmergedNodes.erase(mUnmergedNodes.begin() + i);

      delete node; // NOLINT(cppcoreguidelines-owning-memory): TODO where does it get created?
    } else {
      ++i;
    }
  }

  // copy any nodes that where not merged to mUnmergedNodes
  if (unmerged > 0) {
    // Store unmerged nodes together with the current frame number.
    // Attempts to merge these into the tree will be made until their age
    // exceeds maxUnmergedAge (see mergeUnmerged).
    for (auto const& node : mergeNodes) {
      if (node) {
        mUnmergedNodes.emplace_back(node, mFrameCount);
        --unmerged;
      }
    }
  }

  if (merged > 0 || unmerged > 0) {
#if !defined(NDEBUG) && !defined(VISTAPLANET_NO_VERBOSE)
    vstr::outi() << "[TreeManager::merge] nodes merged/unmerged " << merged << " / " << unmerged
                 << std::endl;
#endif
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TileQuadTree* TreeManager::getTree() {
  return &mTree;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::shared_ptr<GLResources> const& TreeManager::getGLResources() const {
  return mGLResources;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void TreeManager::setFrameCount(int frameCount) {
  mFrameCount = frameCount;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::lodbodies
