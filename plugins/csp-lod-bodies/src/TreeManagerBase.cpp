////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "TreeManagerBase.hpp"

#include "PlanetParameters.hpp"
#include "RenderData.hpp"
#include "TileSource.hpp"
#include "TileTextureArray.hpp"

#include <VistaBase/VistaStreamUtils.h>

#include <utility>

namespace csp::lodbodies {

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace {

////////////////////////////////////////////////////////////////////////////////////////////////////

// The following constants could be made member variables of TreeManagerBase
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
struct TreeManagerBase::AgeLess {
  explicit AgeLess(int frame);

  bool operator()(RDMapValue const* lhs, RDMapValue const* rhs) const;

  int mFrame;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

/* explicit */
TreeManagerBase::AgeLess::AgeLess(int frame)
    : mFrame(frame) {
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool TreeManagerBase::AgeLess::operator()(RDMapValue const* lhs, RDMapValue const* rhs) const {
  // sort by age (frames since last use), in case of a tie use
  // tile level - this ensure that child nodes are always sorted
  // after parent nodes

  int ageLHS = lhs->second->getAge(mFrame);
  int ageRHS = rhs->second->getAge(mFrame);

  if (ageLHS == ageRHS) {
    return lhs->first.level() < rhs->first.level();
  }

  return ageLHS < ageRHS;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

/* explicit */
TreeManagerBase::NodeAge::NodeAge(TileNode* node, int frame)
    : mNode(node)
    , mFrame(frame) {
}

////////////////////////////////////////////////////////////////////////////////////////////////////

/* explicit */
TreeManagerBase::TreeManagerBase(
    PlanetParameters const& params, std::shared_ptr<GLResources> glResources)
    : mParams(&params)
    , mGlMgr(std::move(glResources))
    , mSrc()
    , mFrameCount(0)
    , mAsyncLoading(true) {
  mRdMap.reserve(preAllocNodeCount);
  mAgeStore.reserve(preAllocNodeCount);

  mUnmergedNodes.reserve(preAllocIONodeCount);
  mLoadedNodes.reserve(preAllocIONodeCount);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

/* virtual */
TreeManagerBase::~TreeManagerBase() = default;

////////////////////////////////////////////////////////////////////////////////////////////////////

void TreeManagerBase::setSource(TileSource* src) {
  // remove all existing nodes
  std::unique_lock<std::mutex> lck(mLoadedMtx);
  clear();
  mSrc = src;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TileSource* TreeManagerBase::getSource() const {
  return mSrc;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void TreeManagerBase::request(std::vector<TileId> const& tileIds) {
  // for each requested tile, check if it is already in the mPendingTiles
  // set (those are tiles that have already been requesed from the tile
  // source), otherwise put the tile in mPendingTiles and ask the source to
  // load the tile.
  // In the case of async loading, register @c onNodeLoaded as the callback
  // that the source invokes when the tile is ready.
  std::unique_lock<std::mutex> lck(mLoadedMtx);

  auto iIt  = tileIds.begin();
  auto iEnd = tileIds.end();

  for (; iIt != iEnd; ++iIt) {
    if (mPendingTiles.count(*iIt) == 0) {
      mPendingTiles.insert(*iIt);

      if (mAsyncLoading) {
#if (BOOST_VERSION / 100) % 1000 < 60
        mSrc->loadTileAsync(iIt->level(), iIt->patchIdx(),
            std::bind(&TreeManagerBase::onNodeLoaded, this, _1, _2, _3, _4));
#else
        mSrc->loadTileAsync(iIt->level(), iIt->patchIdx(),
            [this](auto a, auto b, auto c, auto d) { onNodeLoaded(a, b, c, d); });
#endif
      } else {
        TileNode* node = mSrc->loadTile(iIt->level(), iIt->patchIdx());
        onNodeLoaded(mSrc, iIt->level(), iIt->patchIdx(), node);
      }
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void TreeManagerBase::update() {
  // remove unused nodes - do this before the merge to free up resources
  // that can then be consumed by newly loaded ones.
  prune();

  // insert new nodes
  merge();

  // upload tiles to GPU
  getTileTextureArray().processQueue(20);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void TreeManagerBase::clear() {
  mPendingTiles.clear();
  mLoadedNodes.clear();

  auto rdIt  = mRdMap.begin();
  auto rdEnd = mRdMap.end();

  for (; rdIt != rdEnd; ++rdIt) {
    releaseResources(rdIt->second);
  }

  mRdMap.clear();
  mAgeStore.clear();

  for (int i = 0; i < TileQuadTree::sNumRoots; ++i) {
    mTree.setRoot(i, nullptr);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

RenderData const* TreeManagerBase::findRData(TileNode const* node) const {
  return findRData(node->getTileId());
}

////////////////////////////////////////////////////////////////////////////////////////////////////

RenderData* TreeManagerBase::findRData(TileNode const* node) {
  return findRData(node->getTileId());
}

////////////////////////////////////////////////////////////////////////////////////////////////////

RenderData const* TreeManagerBase::findRData(TileId const& tileId) const {
  RenderData const* result = nullptr;
  auto              rdIt   = mRdMap.find(tileId);

  if (rdIt != mRdMap.end()) {
    result = rdIt->second;
  }

  return result;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

RenderData* TreeManagerBase::findRData(TileId const& tileId) {
  RenderData* result = nullptr;
  auto        rdIt   = mRdMap.find(tileId);

  if (rdIt != mRdMap.end()) {
    result = rdIt->second;
  }

  return result;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::size_t TreeManagerBase::getNodeCount() const {
  return mRdMap.size();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::size_t TreeManagerBase::getNodeCountGPU() const {
  return getTileTextureArray().getUsedLayerCount();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void TreeManagerBase::onNodeLoaded(
    TileSource* source, int level, glm::int64 patchIdx, TileNode* node) {
  std::unique_lock<std::mutex> lck(mLoadedMtx);
  if (node && source == mSrc) {
    // Only add node to list of loaded nodes, actual insertion into the
    // quad-tree is done in merge().
    // This ensures that the tree is not modified at unpredictable moments
    // in time (for example while a traversal is in progress).

    mLoadedNodes.push_back(node);
  } else {
    // source has changed or loading failed, discard node
    mPendingTiles.erase(TileId(level, patchIdx));
    delete node; // NOLINT(cppcoreguidelines-owning-memory): TODO where does it get created?
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void TreeManagerBase::onNodeInserted(TileNode* node) {
  RenderData* rdata = allocateRenderData(node);

  if (node->getParent()) {
    RenderData* rdataP = findRData(node->getParent());
    assert(rdataP != nullptr);

    rdata->setLastFrame(rdataP->getLastFrame());
  }

  auto res = mRdMap.insert(RDMapValue(node->getTileId(), rdata));
  assert(res.second);

  getTileTextureArray().allocateGPU(rdata);
  mAgeStore.push_back(&(*res.first));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void TreeManagerBase::releaseResources(RenderData* rdata) {
  getTileTextureArray().releaseGPU(rdata);
  releaseRenderData(rdata);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void TreeManagerBase::prune() {
  // sort by age, oldest nodes at the back
  std::sort(mAgeStore.begin(), mAgeStore.end(), AgeLess(mFrameCount));
  int count = 0;

  while (!mAgeStore.empty()) {
    RDMapValue* value = mAgeStore.back();

    // remove unnused nodes, but never root nodes
    if (value->second->getAge(mFrameCount) > maxNodeAge && value->first.level() > 0) {
      TileNode* node = value->second->getNode();

      releaseResources(value->second);

      if (!removeNode(&mTree, node)) {
        vstr::errp() << "[TreeManagerBase::prune] [" << mName << "] Failed to remove node "
                     << value->first << " @ " << node << "!" << std::endl;
      }

      // remove entries for node from internal data structures
      mRdMap.erase(value->first);
      mAgeStore.pop_back();
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
    vstr::outi() << "[TreeManagerBase::prune] [" << mName << "] nodes removed/kept " << count
                 << " / " << mRdMap.size() << std::endl;
#endif
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void TreeManagerBase::merge() {
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
      mPendingTiles.erase(node->getTileId());
      onNodeInserted(node);

      ++merged;
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
    vstr::outi() << "[TreeManagerBase::merge] [" << mName << "] nodes merged/unmerged " << merged
                 << " / " << unmerged << std::endl;
#endif
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TileQuadTree* TreeManagerBase::getTree() {
  return &mTree;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::string const& TreeManagerBase::getName() const {
  return mName;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void TreeManagerBase::setName(std::string const& name) {
  mName = name;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

int TreeManagerBase::getFrameCount() const {
  return mFrameCount;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void TreeManagerBase::setFrameCount(int frameCount) {
  mFrameCount = frameCount;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TileTextureArray& TreeManagerBase::getTileTextureArray() const {
  return (*mGlMgr)[mSrc->getDataType()];
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::lodbodies
