////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "LODVisitor.hpp"

#include "PlanetParameters.hpp"
#include "TileTextureArray.hpp"
#include "TreeManager.hpp"
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
    BoundingBox<double> const& tb, TreeManager* treeMgr) {
  assert(treeMgr != nullptr);

  // Get minimum height of all base patches (needed for radius of proxy culling sphere)
  auto minHeight(std::numeric_limits<float>::max());
  for (int i(0); i < TileQuadTree::sNumRoots; ++i) {
    auto* tile = treeMgr->getTree()->getRoot(i);
    minHeight  = std::min(minHeight, tile->getMinMaxPyramid()->getMin());
  }

  double dProxyRadius = std::min(params->mRadii.x, std::min(params->mRadii.y, params->mRadii.z)) +
                        (minHeight * params->mHeightScale);

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
bool childrenAvailable(TileNode* node, TreeManager* treeMgr) {
  for (int i = 0; i < 4; ++i) {
    TileNode* child = node->getChild(i);

    // child is not loaded -> can not refine
    if (child == nullptr) {
      return false;
    }

    auto dem = child->getTileData(TileDataType::eElevation);
    auto img = child->getTileData(TileDataType::eColor);

    // child is not on GPU -> can not refine
    if (dem->getTexLayer() < 0 || (img && img->getTexLayer() < 0)) {
      return false;
    }
  }

  return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace

////////////////////////////////////////////////////////////////////////////////////////////////////

/* explicit */
LODVisitor::LODVisitor(PlanetParameters const& params, TreeManager* treeMgr)
    : TileVisitor<LODVisitor>(treeMgr->getTree())
    , mParams(&params)
    , mTreeMgr(treeMgr)
    , mViewport()
    , mMatVM()
    , mMatP()
    , mLodData()
    , mCullData()
    , mStackTop(-1)
    , mFrameCount(0)
    , mUpdateLOD(true)
    , mUpdateCulling(true) {

  mLoadNodes.reserve(PreAllocSize);
  mRenderNodes.reserve(PreAllocSize);

  mStack.resize(sMaxStackDepth);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void LODVisitor::queueRecomputeTileBounds() {
  mRecomputeTileBounds = true;
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
  mLoadNodes.clear();
  mRenderNodes.clear();
  mStackTop = -1;

  // make sure root nodes are present
  for (int i = 0; i < TileQuadTree::sNumRoots; ++i) {
    if (!mTree->getRoot(i)) {
      mLoadNodes.emplace_back(0, i);
      result = false;
    }
  }

  return result;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void LODVisitor::postTraverse() {
  mRecomputeTileBounds = false;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool LODVisitor::preVisitRoot(TileId const& tileId) {
  StateBase& state = getState();

  // fetch tile data for visited node and mark as used in this frame
  if (state.mNode) {
    state.mNode->setLastFrame(mFrameCount);
  }

  return visitNode(tileId);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool LODVisitor::preVisit(TileId const& tileId) {

  StateBase& state = getState();

  // fetch tile data for visited node and mark as used in this frame
  if (state.mNode) {
    state.mNode->setLastFrame(mFrameCount);
  }

  return visitNode(tileId);
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

  StateBase& state = getState();

  if (state.mNode && (!state.mNode->hasBounds() || mRecomputeTileBounds)) {
    auto bounds = calcTileBounds(*state.mNode, mParams->mRadii, mParams->mHeightScale);
    state.mNode->setBounds(bounds);
  }

  bool result  = false;
  bool visible = testVisible(tileId);

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
  StateBase& state = getState();

  // test if nodes can be refined
  bool hasChildren = state.mNode ? childrenAvailable(state.mNode, mTreeMgr) : false;

  // request to load missing children
  if (!hasChildren) {
    addLoadChildren(state.mNode);
    drawLevel();
    return false;
  }

  return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void LODVisitor::addLoadChildren(TileNode* node) {
  if (node && node->getLevel() < mParams->mMaxLevel) {
    TileId const& tileId = node->getTileId();

    for (int i = 0; i < 4; ++i) {
      if (!node->getChild(i)) {
        mLoadNodes.push_back(HEALPix::getChildTileId(tileId, i));
      } else {
        // mark child as used to avoid it being removed while waiting
        // for its siblings to be loaded
        node->getChild(i)->setLastFrame(mFrameCount);
      }
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool LODVisitor::testVisible(TileId const& tileId) {
  bool       result = false;
  StateBase& state  = getState();

  BoundingBox<double> const& tb = state.mNode->getBounds();

  result = testInFrustum(mCullData.mFrustumMS, tb);

  if (result) {
    result = testFrontFacing(mCullData.mCamPos, mParams, tb, mTreeMgr);
  }

  return result;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool LODVisitor::testNeedRefine(TileId const& tileId) {
  bool       result = false;
  StateBase& state  = getState();

  if (state.mNode) {
    BoundingBox<double> tb = state.mNode->getBounds();

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
  }

  return result;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void LODVisitor::drawLevel() {
  StateBase& state = getState();

  // check node is available (either for this level or highest resolution
  // currently loaded) and has data
  assert(state.mNode);

  mRenderNodes.push_back(state.mNode);
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

std::vector<TileId> const& LODVisitor::getLoadNodes() const {
  return mLoadNodes;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::vector<TileNode*> const& LODVisitor::getRenderNodes() const {
  return mRenderNodes;
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

} // namespace csp::lodbodies
