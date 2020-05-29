////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "LODVisitor.hpp"

#include "PlanetParameters.hpp"
#include "RenderDataDEM.hpp"
#include "RenderDataImg.hpp"
#include "TileTextureArray.hpp"
#include "TreeManagerBase.hpp"
#include "logger.hpp"

#include <VistaBase/VistaStreamUtils.h>
#include <glm/gtc/matrix_inverse.hpp>

namespace csp::lodbodies {

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace {
// Initially reserve storage for this many entries in the lists produced
// by LODVisitor (mLoadDEM, mLoadIMG, mRenderDEM, mRenderIMG).
// The lists are still grown as needed, but this reduces the number of
// re-allocations.
std::size_t const PreAllocSize = 200;

////////////////////////////////////////////////////////////////////////////////////////////////////

// Returns if the tile bounds @a tb intersect the @a frustum.
// For each plane of the @a frustum determine if any corner of the
// bounding box is inside the plane's halfspace. If all corners are
// outside one halfspace the bounding box is outside the frustum
// and the algorithm stops early.
// TODO There is potential for optimization here, the paper
// "Optimized View Frustum Culling - Algorithms for Bounding Boxes"
// http://www.cse.chalmers.se/~uffe/vfc_bbox.pdf
// contains ideas (for example how to avoid testing all 8 corners).
bool testInFrustum(Frustum const& frustum, BoundingBox<double> const& tb) {
  bool result = true;

  glm::dvec3 const& tbMin = tb.getMin();
  glm::dvec3 const& tbMax = tb.getMax();

  // 8 corners of tile's bounding box
  std::array<glm::dvec3, 8> tbPnts = {
      {glm::dvec3(tbMin[0], tbMin[1], tbMin[2]), glm::dvec3(tbMax[0], tbMin[1], tbMin[2]),
          glm::dvec3(tbMax[0], tbMin[1], tbMax[2]), glm::dvec3(tbMin[0], tbMin[1], tbMax[2]),

          glm::dvec3(tbMin[0], tbMax[1], tbMin[2]), glm::dvec3(tbMax[0], tbMax[1], tbMin[2]),
          glm::dvec3(tbMax[0], tbMax[1], tbMax[2]), glm::dvec3(tbMin[0], tbMax[1], tbMax[2])}};

  // loop over planes of frustum
  auto pIt  = frustum.getPlanes().begin();
  auto pEnd = frustum.getPlanes().end();

  for (std::size_t i = 0; pIt != pEnd; ++pIt, ++i) {
    glm::dvec3 const normal(*pIt);
    double const     d       = -(*pIt)[3];
    bool             outside = true;

    // test if any BB corner is inside the halfspace defined
    // by the current plane
    for (auto const& tbPnt : tbPnts) {
      if (glm::dot(normal, tbPnt) >= d) {
        // corner j is inside - stop testing
        outside = false;
        break;
      }
    }

    // if all corners are outside the halfspace, the bounding box
    // is outside - stop testing
    if (outside) {
      result = false;
      break;
    }
  }

  return result;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Returns true if one the eight tile bbox corner points is not occluded by a proxy sphere.
// Culls tiles behind the horizon.
bool testFrontFacing(glm::dvec3 const& camPos, PlanetParameters const* params,
    BoundingBox<double> const& tb, TreeManagerBase* treeMgrDEM) {
  assert(treeMgrDEM != nullptr);

  // Get minimum height of all base patches (needed for radius of proxy culling sphere)
  auto minHeight(std::numeric_limits<float>::max());
  for (int i(0); i < TileQuadTree::sNumRoots; ++i) {
    auto*       tile       = treeMgrDEM->getTree()->getRoot(i)->getTile();
    auto const& castedTile = dynamic_cast<Tile<float> const&>(*tile);
    minHeight              = std::min(minHeight, castedTile.getMinMaxPyramid()->getMin());
  }

  double dScaledPolarRadius = params->mPolarRadius + (minHeight * params->mHeightScale);
  double dProxyRadius =
      std::min(dScaledPolarRadius, params->mEquatorialRadius + (minHeight * params->mHeightScale));

  glm::dvec3 const& tbMin = tb.getMin();
  glm::dvec3 const& tbMax = tb.getMax();

  // 8 corners of tile's bounding box
  std::array<glm::dvec3, 8> tbPnts = {
      {glm::dvec3(tbMin[0], tbMin[1], tbMin[2]), glm::dvec3(tbMax[0], tbMin[1], tbMin[2]),
          glm::dvec3(tbMax[0], tbMin[1], tbMax[2]), glm::dvec3(tbMin[0], tbMin[1], tbMax[2]),

          glm::dvec3(tbMin[0], tbMax[1], tbMin[2]), glm::dvec3(tbMax[0], tbMax[1], tbMin[2]),
          glm::dvec3(tbMax[0], tbMax[1], tbMax[2]), glm::dvec3(tbMin[0], tbMax[1], tbMax[2])}};

  // Simple ray-sphere intersection test for every corner point
  for (auto const& tbPnt : tbPnts) {
    double     dRayLength = glm::length(tbPnt - camPos);
    glm::dvec3 vRayDir    = (tbPnt - camPos) / dRayLength;
    double     b          = glm::dot(camPos, vRayDir);
    double     c          = glm::dot(camPos, camPos) - dProxyRadius * dProxyRadius;
    double     fDet       = b * b - c;
    // No intersection between corner and camera position: Tile visible!:
    if (fDet < 0.0) {
      return true;
    }

    fDet = std::sqrt(fDet);
    // Both intersection points are behind the camera but tile is in front
    // (presumes tiles to be frustum culled already!!!)
    // E.g. While travelling in a deep crater and looking above
    if ((-b - fDet) < 0.0 && (-b + fDet) < 0.0) {
      return true;
    }

    // Tile in front of planet:
    if (dRayLength < -b - fDet) {
      return true;
    }
  }

  return false;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Tests if @a node can be refined, which is the case if all 4
// children are present and uploaded to the GPU.
bool childrenAvailable(TileNode* node, TreeManagerBase* treeMgr) {
  for (int i = 0; i < 4; ++i) {
    TileNode* child = node->getChild(i);

    // child is not loaded -> can not refine
    if (child == nullptr) {
      return false;
    }

    RenderData* rd = treeMgr->findRData(child);

    // child is not on GPU -> can not refine
    if (!rd || rd->getTexLayer() < 0) {
      return false;
    }
  }

  return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Returns the @c RenderDataDEM for tile @a tileId or (if that is
// not available) the closest parent's render data - only @c RenderDataDEM
// that are marked as @c RenderDataDEM::Flags::eRender are considered. If none
// is found but there is renderdata for the tile @a tileId, this will be
// returned.
RenderDataDEM* findParentRData(TreeManagerBase* treeMgr, TileId tileId) {
  auto*          rdata     = treeMgr->find<RenderDataDEM>(tileId);
  RenderDataDEM* origRdata = rdata;

  while (!rdata || !rdata->testFlag(RenderDataDEM::Flags::eRender)) {
    if (tileId.level() == 0) {
      break;
    }

    tileId = HEALPix::getParentTileId(tileId);
    rdata  = treeMgr->find<RenderDataDEM>(tileId);
  }

  assert(rdata != nullptr);
  // none of the parents is rendered - return the actual thing
  if (origRdata && !rdata->testFlag(RenderDataDEM::Flags::eRender)) {
    return origRdata;
  }

  return rdata;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace

////////////////////////////////////////////////////////////////////////////////////////////////////

/* explicit */
LODVisitor::LODVisitor(
    PlanetParameters const& params, TreeManagerBase* treeMgrDEM, TreeManagerBase* treeMgrIMG)
    : TileVisitor<LODVisitor>(treeMgrDEM ? treeMgrDEM->getTree() : nullptr,
          treeMgrIMG ? treeMgrIMG->getTree() : nullptr)
    , mParams(&params)
    , mTreeMgrDEM(nullptr)
    , mTreeMgrIMG(nullptr)
    , mViewport()
    , mMatVM()
    , mMatP()
    , mLodData()
    , mCullData()
    , mStackTop(-1)
    , mFrameCount(0)
    , mUpdateLOD(true)
    , mUpdateCulling(true) {
  setTreeManagerDEM(treeMgrDEM);
  setTreeManagerIMG(treeMgrIMG);

  mStack.resize(sMaxStackDepth);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void LODVisitor::setTreeManagerDEM(TreeManagerBase* treeMgr) {
  // unset tree from OLD tree manager
  if (mTreeMgrDEM) {
    setTreeDEM(nullptr);
  }

  mStack.clear();
  mStack.resize(sMaxStackDepth);

  mTreeMgrDEM = treeMgr;

  // set tree from NEW tree manager
  if (mTreeMgrDEM) {
    setTreeDEM(mTreeMgrDEM->getTree());
    mLoadDEM.reserve(PreAllocSize);
    mRenderDEM.reserve(PreAllocSize);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void LODVisitor::setTreeManagerIMG(TreeManagerBase* treeMgr) {
  // unset tree from OLD tree manager
  if (mTreeMgrIMG) {
    setTreeIMG(nullptr);
  }

  mTreeMgrIMG = treeMgr;

  // set tree from NEW tree manager
  if (mTreeMgrIMG) {
    setTreeIMG(mTreeMgrIMG->getTree());
    mLoadIMG.reserve(PreAllocSize);
    mRenderIMG.reserve(PreAllocSize);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool LODVisitor::preTraverse() {
  bool result = true;

  // update derived matrices from mMatP, mMatVM
  if (mUpdateLOD) {
    mLodData.mMatVM = mMatVM;
    mLodData.mMatP  = mMatP;
    mLodData.mFrustumES.setFromMatrix(mMatP);
    mLodData.mViewport = mViewport;
  }

  if (mUpdateCulling) {
    mCullData.mFrustumMS.setFromMatrix(mMatP * mMatVM);
    mCullData.mMatN   = glm::inverseTranspose(glm::f64mat3x3(mMatVM));
    auto v4CamPos     = glm::inverse(mMatVM)[3];
    mCullData.mCamPos = glm::dvec3(v4CamPos[0], v4CamPos[1], v4CamPos[2]);
  }

  // clear load/render lists
  mLoadDEM.clear();
  mLoadIMG.clear();
  mRenderDEM.clear();
  mRenderIMG.clear();
  mStackTop = -1;

  // make sure root nodes are present
  for (int i = 0; i < TileQuadTree::sNumRoots; ++i) {
    if (mTreeDEM) {
      if (!mTreeDEM->getRoot(i)) {
        mLoadDEM.emplace_back(0, i);
        result = false;
      }
    }

    if (mTreeIMG) {
      if (!mTreeIMG->getRoot(i)) {
        mLoadIMG.emplace_back(0, i);
        result = false;
      }
    }
  }

  return result;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void LODVisitor::postTraverse() {
  // Determine edges with LOD change
  // For each edge the level difference is stored and the RenderDataDEM
  // object for the neighbour. If the level of the neighbour is lower,
  // the RenderDataDEM of the parent (or the parent's parent or ...) will
  // be stored.
  for (auto* rd : mRenderDEM) {
    auto*         rdDEM  = dynamic_cast<RenderDataDEM*>(rd);
    TileId const& tileId = rd->getNode()->getTileId();

    auto nIds = HEALPix::getNeighbourIds(tileId);

    // base patch -- need to check all 4 neighbours
    RenderDataDEM* rdNE = findParentRData(mTreeMgrDEM, nIds[0]);
    RenderDataDEM* rdNW = findParentRData(mTreeMgrDEM, nIds[1]);
    RenderDataDEM* rdSW = findParentRData(mTreeMgrDEM, nIds[2]);
    RenderDataDEM* rdSE = findParentRData(mTreeMgrDEM, nIds[3]);

    if (rdNE) {
      int delta = 1;
      if (rdNE->testFlag(RenderDataDEM::Flags::eRender)) {
        delta = rdNE->getLevel() - tileId.level();
      }
      rdDEM->setEdgeDelta(0, delta);
      rdDEM->setEdgeRData(0, rdNE);
    }
    if (rdNW) {
      int delta = 1;
      if (rdNW->testFlag(RenderDataDEM::Flags::eRender)) {
        delta = rdNW->getLevel() - tileId.level();
      }
      rdDEM->setEdgeDelta(1, delta);
      rdDEM->setEdgeRData(1, rdNW);
    }
    if (rdSW) {
      int delta = 1;
      if (rdSW->testFlag(RenderDataDEM::Flags::eRender)) {
        delta = rdSW->getLevel() - tileId.level();
      }
      rdDEM->setEdgeDelta(2, delta);
      rdDEM->setEdgeRData(2, rdSW);
    }
    if (rdSE) {
      int delta = 1;
      if (rdSE->testFlag(RenderDataDEM::Flags::eRender)) {
        delta = rdSE->getLevel() - tileId.level();
      }
      rdDEM->setEdgeDelta(3, delta);
      rdDEM->setEdgeRData(3, rdSE);
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool LODVisitor::preVisitRoot(TileId const& tileId) {
  LODState& state = getLODState();

  // track highest resolution nodes in this sub tree (in case there is
  // higher resolution image data than DEM data).
  state.mLastDEM  = nullptr;
  state.mLastIMG  = nullptr;
  state.mMaxLevel = 0;

  // fetch RenderDataDEM for visited node and mark as used in this frame
  if (mTreeMgrDEM && state.mNodeDEM) {
    auto* rd     = mTreeMgrDEM->find<RenderDataDEM>(state.mNodeDEM);
    state.mRdDEM = rd;
    state.mRdDEM->setLastFrame(mFrameCount);
  } else {
    state.mRdDEM = nullptr;
  }

  // fetch RenderDataImg for visited node and mark as used in this frame
  if (mTreeMgrIMG && state.mNodeIMG) {
    auto* rd     = mTreeMgrIMG->find<RenderDataImg>(state.mNodeIMG);
    state.mRdIMG = rd;
    state.mRdIMG->setLastFrame(mFrameCount);
  } else {
    state.mRdIMG = nullptr;
  }

  return visitNode(tileId);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void LODVisitor::postVisitRoot(TileId const& /*tileId*/) {
  // nothing to do
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool LODVisitor::preVisit(TileId const& tileId) {

  LODState& state  = getLODState();
  LODState& stateP = getLODState(tileId.level() - 1); // parent state

  // track highest resolution nodes that can not be refined further (e.g.
  // because not all 4 children are loaded) - these are NULL if parent nodes
  // so far can all be refined
  state.mLastDEM  = stateP.mLastDEM;
  state.mLastIMG  = stateP.mLastIMG;
  state.mMaxLevel = stateP.mMaxLevel;

  // fetch RenderDataDEM for visited node and mark as used in this frame
  if (mTreeMgrDEM && !state.mLastDEM && state.mNodeDEM) {
    auto* rd     = mTreeMgrDEM->find<RenderDataDEM>(state.mNodeDEM);
    state.mRdDEM = rd;
    state.mRdDEM->setLastFrame(mFrameCount);
  } else {
    // copy value from parent state to ensure this matches state.mLastDEM
    state.mRdDEM = stateP.mRdDEM;
  }

  // fetch RenderDataImg for visited node and mark as used in this frame
  if (mTreeMgrIMG && !state.mLastIMG && state.mNodeIMG) {
    auto* rd     = mTreeMgrIMG->find<RenderDataImg>(state.mNodeIMG);
    state.mRdIMG = rd;
    state.mRdIMG->setLastFrame(mFrameCount);
  } else {
    // copy value from parent state to ensure this matches state.mLastIMG
    state.mRdIMG = stateP.mRdIMG;
  }

  return visitNode(tileId);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void LODVisitor::postVisit(TileId const& /*tileId*/) {
  // nothing to do
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool LODVisitor::visitNode(TileId const& tileId) {
  // Determine if the current node is visible (using the DEM node).
  // This is done by testing for an intersection between the camera
  // frustum and the DEM node's bounding box.
  //
  // If the node is visible:
  //      Determine if the node has sufficient resolution for the current
  //      view: see testNeedRefine() for details on the algorithm.
  //
  //      If node should be refined:
  //          Determine if it is possible to refine the node, see
  //          handleRefine() for details.
  //      Else:
  //          draw this level

  bool result  = false;
  bool visible = testVisible(tileId, mTreeMgrDEM);

  if (visible) {
    // should this node be refined to achieve desired resolution?
    bool needRefine = testNeedRefine(tileId);

    if (needRefine) {
      result = handleRefine(tileId);
    } else {
      // resolution is sufficient
      drawLevel();
    }
  }

  return result;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool LODVisitor::handleRefine(TileId const& /*tileId*/) {
  bool      result  = false;
  LODState& state   = getLODState();
  TileNode* nodeDEM = !state.mLastDEM ? state.mNodeDEM : nullptr;
  TileNode* nodeIMG = !state.mLastIMG ? state.mNodeIMG : nullptr;

  // test if nodes can be refined
  bool childrenDemAvailable = nodeDEM ? childrenAvailable(nodeDEM, mTreeMgrDEM) : false;
  bool childrenImgAvailable = nodeIMG ? childrenAvailable(nodeIMG, mTreeMgrIMG) : false;

  if (mTreeMgrDEM != nullptr && mTreeMgrIMG != nullptr) {
    // DEM and IMG data

    // request to load missing children
    if (!childrenDemAvailable) {
      state.mLastDEM = state.mLastDEM ? state.mLastDEM : state.mNodeDEM;
      addLoadChildrenDEM(nodeDEM);
    }

    if (!childrenImgAvailable) {
      state.mLastIMG = state.mLastIMG ? state.mLastIMG : state.mNodeIMG;
      addLoadChildrenIMG(nodeIMG);
    }

    if (childrenDemAvailable || childrenImgAvailable) {
      // at least one tree (DEM or IMG) can be refined, visit children
      result = true;
    } else {
      // can not refine, draw this level
      drawLevel();
    }
  } else if (mTreeMgrDEM != nullptr && mTreeMgrIMG == nullptr) {
    // DEM data only

    if (childrenDemAvailable) {
      // tree can be refined, visit children
      result = true;
    } else {
      // request to load missing children
      state.mLastDEM = state.mLastDEM ? state.mLastDEM : state.mNodeDEM;
      addLoadChildrenDEM(nodeDEM);

      // can not refine, draw this level
      drawLevel();
    }
  }

  return result;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void LODVisitor::addLoadChildrenDEM(TileNode* node) {
  if (node && !isLeaf(*node)) {
    TileId const& tileId = node->getTileId();

    for (int i = 0; i < 4; ++i) {
      if (!node->getChild(i)) {
        mLoadDEM.push_back(HEALPix::getChildTileId(tileId, i));
      } else {
        // mark child as used to avoid it being removed while waiting
        // for its siblings to be loaded
        RenderData* rd = mTreeMgrDEM->findRData(node->getChild(i));
        rd->setLastFrame(mFrameCount);
      }
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void LODVisitor::addLoadChildrenIMG(TileNode* node) {
  if (node && !isLeaf(*node)) {
    TileId const& tileId = node->getTileId();

    for (int i = 0; i < 4; ++i) {
      if (!node->getChild(i)) {
        mLoadIMG.push_back(HEALPix::getChildTileId(tileId, i));
      } else {
        // mark child as used to avoid it being removed while waiting
        // for its siblings to be loaded
        RenderData* rd = mTreeMgrIMG->findRData(node->getChild(i));
        rd->setLastFrame(mFrameCount);
      }
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool LODVisitor::testVisible(TileId const& tileId, TreeManagerBase* treeMgrDEM) {
  bool      result = false;
  LODState& state  = getLODState();

  if (state.mNodeDEM) {
    // Simple case, there is a DEM node for this level, i.e.
    // DEM resolution is at least as high as IMG resolution.
    // Use the bounds of the DEM node to decide visibility.

    BoundingBox<double> const& tb = state.mRdDEM->getBounds();

    result = testInFrustum(mCullData.mFrustumMS, tb);

    if (result) {
      result = testFrontFacing(mCullData.mCamPos, mParams, tb, treeMgrDEM);
    }

    if (state.mRdIMG && state.mRdIMG->hasBounds()) {
      state.mRdIMG->removeBounds();
    }
  } else {
    BoundingBox<double> tb;

    // Get MinMaxPyramid of last known DEM tile
    auto* tileBaseDEM = state.mLastDEM->getTile();
    if (tileBaseDEM->getDataType() == TileDataType::eFloat32) {
      auto* tileDEM = dynamic_cast<Tile<float>*>(tileBaseDEM);
      if (auto* pyr = tileDEM->getMinMaxPyramid()) {

        auto  lvl = tileId.level();
        float minHeight(0);
        float maxHeight(0);
        // Level difference to the last DEM tile:
        auto levelDiff = std::min(7, lvl - state.mLastDEM->getLevel());

        // Collect child indices of all (parent) IMG nodes without DEM
        // They define the location of the IMG tile in the last DEM tile
        std::vector<int> quadrants;
        auto             cur_tileId = tileId;
        for (int i(0); i < levelDiff; ++i) {
          quadrants.insert(
              quadrants.begin(), HEALPix::getChildIdxAtLevel(cur_tileId, cur_tileId.level()));
          cur_tileId = HEALPix::getParentTileId(cur_tileId);
        }

        // Get min and max height value from the coarser DEM tile
        minHeight = pyr->getMin(quadrants);
        maxHeight = pyr->getMax(quadrants);

        // Calculate an optimistic bounding box from the the height values in range of the
        // IMG tile
        tb = csp::lodbodies::calcTileBounds(minHeight, maxHeight, lvl, tileId.patchIdx(),
            mParams->mEquatorialRadius, mParams->mPolarRadius, mParams->mHeightScale);

        // Save for renderBounds() in TileRenderer
        state.mRdIMG->setBounds(tb);
      }
    } else {
      logger().error("Failed to test visibility of Tile: Unknown tile template type!");
    }

    result = testInFrustum(mCullData.mFrustumMS, tb);

    if (result) {
      result = testFrontFacing(mCullData.mCamPos, mParams, tb, treeMgrDEM);
    }
    // result = true;
  }

  return result;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool LODVisitor::testNeedRefine(TileId const& tileId) {
  bool      result = false;
  LODState& state  = getLODState();

  if (state.mNodeDEM || state.mRdIMG->hasBounds()) {
    BoundingBox<double> tb;

    if (state.mNodeDEM) {
      // simple case, there is a DEM node for this level, i.e.
      // DEM resolution is at least as high as IMG resolution
      tb = state.mRdDEM->getBounds();
    } else {
      // use calculated bounds based on minMaxPyramid
      tb = state.mRdIMG->getBounds();
    }

    glm::dvec3 const& tbMin    = tb.getMin();
    glm::dvec3 const& tbMax    = tb.getMax();
    glm::dvec3        tbCenter = 0.5 * (tbMin + tbMax);

    // A tile is refined if the solid angle it occupies when seen from the
    // camera is above a given threshold. To estimate the solid angle, the
    // angles between the vector from the camera to the bounding box center
    // and all vectors from the camera to all eight corners of the bounding
    // box are calculated and the maximum of those is taken.

    // 8 corners of tile's bounding box
    std::array<glm::dvec3, 8> tbDirs = {
        {glm::normalize(glm::dvec3(tbMin.x, tbMin.y, tbMin.z) - mCullData.mCamPos),
            glm::normalize(glm::dvec3(tbMax.x, tbMin.y, tbMin.z) - mCullData.mCamPos),
            glm::normalize(glm::dvec3(tbMax.x, tbMin.y, tbMax.z) - mCullData.mCamPos),
            glm::normalize(glm::dvec3(tbMin.x, tbMin.y, tbMax.z) - mCullData.mCamPos),
            glm::normalize(glm::dvec3(tbMin.x, tbMax.y, tbMin.z) - mCullData.mCamPos),
            glm::normalize(glm::dvec3(tbMax.x, tbMax.y, tbMin.z) - mCullData.mCamPos),
            glm::normalize(glm::dvec3(tbMax.x, tbMax.y, tbMax.z) - mCullData.mCamPos),
            glm::normalize(glm::dvec3(tbMin.x, tbMax.y, tbMax.z) - mCullData.mCamPos)}};

    glm::dvec3 centerDir = glm::normalize(tbCenter - mCullData.mCamPos);

    double maxAngle(0.0);

    for (auto& tbDir : tbDirs) {
      maxAngle = std::max(std::acos(std::min(1.0, glm::dot(tbDir, centerDir))), maxAngle);
    }

    // calculate field of view
    double fov =
        std::max(mLodData.mFrustumES.getHorizontalFOV(), mLodData.mFrustumES.getVerticalFOV());

    double ratio = maxAngle / fov * mParams->mLodFactor;

    result = ratio > 10.0;

    if (mParams->mMinLevel > tileId.level()) {
      result = true;
    }

    // estimate how many more levels are necessary to achieve desired
    // lod factor - used for the case below (no DEM node for level)
    double const deltaLvl = std::max(0.0, std::ceil(std::log(ratio) / std::log(4.0)));
    state.mMaxLevel       = static_cast<int>(tileId.level() + deltaLvl);
  }

  return result;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void LODVisitor::drawLevel() {
  LODState& state = getLODState();

  if (mTreeMgrDEM) {
    // check node is available (either for this level or highest resolution
    // currently loaded) and has RenderDataDEM
    assert(state.mLastDEM || state.mNodeDEM);
    assert(state.mRdDEM);

    state.mRdDEM->addFlag(RenderDataDEM::Flags::eRender);
    mRenderDEM.push_back(state.mRdDEM);
  }

  if (mTreeMgrIMG) {
    // check node is available (either for this level or highest resolution
    // currently loaded) and has RenderDataIMG
    assert(state.mLastIMG || state.mNodeIMG);
    assert(state.mRdIMG);

    mRenderIMG.push_back(state.mRdIMG);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TreeManagerBase* LODVisitor::getTreeManagerDEM() const {
  return mTreeMgrDEM;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TreeManagerBase* LODVisitor::getTreeManagerIMG() const {
  return mTreeMgrIMG;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

int LODVisitor::getFrameCount() const {
  return mFrameCount;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void LODVisitor::setFrameCount(int frameCount) {
  mFrameCount = frameCount;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

glm::ivec4 const& LODVisitor::getViewport() const {
  return mViewport;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void LODVisitor::setViewport(glm::ivec4 const& vp) {
  mViewport = vp;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

glm::dmat4 const& LODVisitor::getModelview() const {
  return mMatVM;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void LODVisitor::setModelview(glm::dmat4 const& m) {
  mMatVM = m;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

glm::dmat4 const& LODVisitor::getProjection() const {
  return mMatP;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void LODVisitor::setProjection(glm::dmat4 const& m) {
  mMatP = m;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void LODVisitor::setUpdateLOD(bool enable) {
  mUpdateLOD = enable;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool LODVisitor::getUpdateLOD() const {
  return mUpdateLOD;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void LODVisitor::setUpdateCulling(bool enable) {
  mUpdateCulling = enable;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool LODVisitor::getUpdateCulling() const {
  return mUpdateCulling;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::vector<TileId> const& LODVisitor::getLoadDEM() const {
  return mLoadDEM;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::vector<TileId> const& LODVisitor::getLoadIMG() const {
  return mLoadIMG;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::vector<RenderData*> const& LODVisitor::getRenderDEM() const {
  return mRenderDEM;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::vector<RenderData*> const& LODVisitor::getRenderIMG() const {
  return mRenderIMG;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void LODVisitor::pushState() {
  mStackTop += 1;

  // check that stack does not overflow
  assert(mStackTop < static_cast<int>(sMaxStackDepth));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void LODVisitor::popState() {
  mStackTop -= 1;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

LODVisitor::StateBase& LODVisitor::getState() {
  // check that stack is valid
  assert(mStackTop >= 0);
  return mStack.at(mStackTop);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

LODVisitor::StateBase const& LODVisitor::getState() const {
  // check that stack is valid
  assert(mStackTop >= 0);
  return mStack.at(mStackTop);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

LODVisitor::LODState& LODVisitor::getLODState(int level /*= -1*/) {
  // check that stack is valid
  assert(mStackTop >= 0);
  return mStack.at(level >= 0 ? level : mStackTop);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

LODVisitor::LODState const& LODVisitor::getLODState(int level /*= -1*/) const {
  // check that stack is valid
  assert(mStackTop >= 0);
  return mStack.at(level >= 0 ? level : mStackTop);
}
////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::lodbodies
