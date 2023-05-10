////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "LODVisitor.hpp"

#include "HEALPix.hpp"
#include "PlanetParameters.hpp"
#include "TileBounds.hpp"
#include "TileTextureArray.hpp"
#include "TreeManager.hpp"
#include "logger.hpp"

#include <VistaBase/VistaStreamUtils.h>
#include <glm/gtc/matrix_inverse.hpp>

namespace csp::lodbodies {

////////////////////////////////////////////////////////////////////////////////////////////////////

/* explicit */
LODVisitor::LODVisitor(PlanetParameters const& params, TreeManager* treeMgr)
    : TileVisitor(treeMgr->getTree())
    , mParams(&params)
    , mTreeMgr(treeMgr)
    , mMatVM()
    , mMatP()
    , mCameraData()
    , mFrameCount(0)
    , mUpdateLOD(true) {
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void LODVisitor::queueRecomputeTileBounds() {
  mRecomputeTileBounds = true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool LODVisitor::preTraverse() {

  mLoadNodes.clear();
  mRenderNodes.clear();

  // Make sure root nodes are loaded.
  for (int i = 0; i < TileQuadTree::sNumRoots; ++i) {
    if (!mTree->getRoot(i)) {
      mLoadNodes.emplace_back(0, i);
      return false;
    }
  }

  // Update derived matrices from mMatP, mMatVM.
  if (mUpdateLOD) {
    mCameraData.mFrustumES.setFromMatrix(mMatP);
    mCameraData.mFrustumMS.setFromMatrix(mMatP * mMatVM);
    mCameraData.mMatN   = glm::inverseTranspose(glm::f64mat3x3(mMatVM));
    auto v4CamPos       = glm::inverse(mMatVM)[3];
    mCameraData.mCamPos = glm::dvec3(v4CamPos[0], v4CamPos[1], v4CamPos[2]);
  }

  // Get minimum height of all base patches (needed for radius of proxy culling sphere).
  auto minHeight(std::numeric_limits<float>::max());
  for (int i(0); i < TileQuadTree::sNumRoots; ++i) {
    auto* tile = mTreeMgr->getTree()->getRoot(i);
    minHeight  = std::min(minHeight, tile->getMinMaxPyramid()->getMin());
  }

  mHorizonCullRadius = std::min(mParams->mRadii.x, std::min(mParams->mRadii.y, mParams->mRadii.z)) +
                       (minHeight * mParams->mHeightScale);

  return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void LODVisitor::postTraverse() {
  mRecomputeTileBounds = false;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool LODVisitor::preVisitRoot(TileNode* root) {
  return visitNode(root);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool LODVisitor::preVisit(TileNode* node) {
  return visitNode(node);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool LODVisitor::visitNode(TileNode* node) {

  // Recompute tile bounds if required.
  if (!node->hasBounds() || mRecomputeTileBounds) {
    auto bounds = calcTileBounds(*node, mParams->mRadii, mParams->mHeightScale);
    node->setBounds(bounds);
  }

  // Mark this node as used to prevent it from being removed.
  node->setLastFrame(mFrameCount);

  // Node is not visible, do not traverse further.
  if (!testInFrustum(node) || !testFrontFacing(node)) {
    return false;
  }

  // If no refinement is required, we can directly render the node and stop the traversal.
  bool needRefine = node->getLevel() < mParams->mMaxLevel && testNeedRefine(node);
  if (!needRefine) {
    mRenderNodes.push_back(node);
    return false;
  }

  // If all children are available, we can continue with the traversal.
  if (node->childrenAvailable()) {
    return true;
  }

  // Else we have to request loading of missing children.
  TileId const& tileId = node->getTileId();

  for (int i = 0; i < 4; ++i) {
    if (!node->getChild(i)) {
      mLoadNodes.push_back(HEALPix::getChildTileId(tileId, i));
    } else {
      // Mark this child as used to avoid it being removed while waiting for its siblings to be
      // loaded.
      node->getChild(i)->setLastFrame(mFrameCount);
    }
  }

  // Finally draw this node until all children are loaded and stop the traversal.
  mRenderNodes.push_back(node);

  return false;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool LODVisitor::testNeedRefine(TileNode* node) const {

  if (mParams->mMinLevel > node->getLevel()) {
    return true;
  }

  glm::dvec3 const& tbMin    = node->getBounds().getMin();
  glm::dvec3 const& tbMax    = node->getBounds().getMax();
  glm::dvec3        tbCenter = 0.5 * (tbMin + tbMax);

  // A tile is refined if the solid angle it occupies when seen from the
  // camera is above a given threshold. To estimate the solid angle, the
  // angles between the vector from the camera to the bounding box center
  // and all vectors from the camera to all eight corners of the bounding
  // box are calculated and the maximum of those is taken.

  // 8 corners of tile's bounding box
  std::array<glm::dvec3, 8> tbDirs = {
      {glm::normalize(glm::dvec3(tbMin.x, tbMin.y, tbMin.z) - mCameraData.mCamPos),
          glm::normalize(glm::dvec3(tbMax.x, tbMin.y, tbMin.z) - mCameraData.mCamPos),
          glm::normalize(glm::dvec3(tbMax.x, tbMin.y, tbMax.z) - mCameraData.mCamPos),
          glm::normalize(glm::dvec3(tbMin.x, tbMin.y, tbMax.z) - mCameraData.mCamPos),
          glm::normalize(glm::dvec3(tbMin.x, tbMax.y, tbMin.z) - mCameraData.mCamPos),
          glm::normalize(glm::dvec3(tbMax.x, tbMax.y, tbMin.z) - mCameraData.mCamPos),
          glm::normalize(glm::dvec3(tbMax.x, tbMax.y, tbMax.z) - mCameraData.mCamPos),
          glm::normalize(glm::dvec3(tbMin.x, tbMax.y, tbMax.z) - mCameraData.mCamPos)}};

  glm::dvec3 centerDir = glm::normalize(tbCenter - mCameraData.mCamPos);

  double maxAngle(0.0);

  for (auto& tbDir : tbDirs) {
    maxAngle = std::max(std::acos(std::min(1.0, glm::dot(tbDir, centerDir))), maxAngle);
  }

  // Calculate field of view.
  double fov =
      std::max(mCameraData.mFrustumES.getHorizontalFOV(), mCameraData.mFrustumES.getVerticalFOV());

  double ratio = maxAngle / fov * mParams->mLodFactor;

  return ratio > 10.0;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void LODVisitor::setFrameCount(int frameCount) {
  mFrameCount = frameCount;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void LODVisitor::setModelview(glm::dmat4 const& m) {
  mMatVM = m;
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

std::vector<TileId> const& LODVisitor::getLoadNodes() const {
  return mLoadNodes;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::vector<TileNode*> const& LODVisitor::getRenderNodes() const {
  return mRenderNodes;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool LODVisitor::testInFrustum(TileNode* node) const {
  BoundingBox<double> const& tb = node->getBounds();

  glm::dvec3 const& tbMin = tb.getMin();
  glm::dvec3 const& tbMax = tb.getMax();

  // 8 corners of tile's bounding box.
  std::array<glm::dvec3, 8> tbPnts = {
      {glm::dvec3(tbMin[0], tbMin[1], tbMin[2]), glm::dvec3(tbMax[0], tbMin[1], tbMin[2]),
          glm::dvec3(tbMax[0], tbMin[1], tbMax[2]), glm::dvec3(tbMin[0], tbMin[1], tbMax[2]),

          glm::dvec3(tbMin[0], tbMax[1], tbMin[2]), glm::dvec3(tbMax[0], tbMax[1], tbMin[2]),
          glm::dvec3(tbMax[0], tbMax[1], tbMax[2]), glm::dvec3(tbMin[0], tbMax[1], tbMax[2])}};

  // Loop over planes of frustum.
  auto pIt  = mCameraData.mFrustumMS.getPlanes().begin();
  auto pEnd = mCameraData.mFrustumMS.getPlanes().end();

  for (std::size_t i = 0; pIt != pEnd; ++pIt, ++i) {
    glm::dvec3 const normal(*pIt);
    double const     d       = -(*pIt)[3];
    bool             outside = true;

    // Test if any BB corner is inside the halfspace defined by the current plane.
    for (auto const& tbPnt : tbPnts) {
      if (glm::dot(normal, tbPnt) >= d) {
        // Corner j is inside - stop testing.
        outside = false;
        break;
      }
    }

    // If all corners are outside the halfspace, the bounding box is outside - stop testing.
    if (outside) {
      return false;
    }
  }

  return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool LODVisitor::testFrontFacing(TileNode* node) const {

  glm::dvec3 const& tbMin = node->getBounds().getMin();
  glm::dvec3 const& tbMax = node->getBounds().getMax();

  // 8 corners of tile's bounding box.
  std::array<glm::dvec3, 8> tbPnts = {
      {glm::dvec3(tbMin[0], tbMin[1], tbMin[2]), glm::dvec3(tbMax[0], tbMin[1], tbMin[2]),
          glm::dvec3(tbMax[0], tbMin[1], tbMax[2]), glm::dvec3(tbMin[0], tbMin[1], tbMax[2]),

          glm::dvec3(tbMin[0], tbMax[1], tbMin[2]), glm::dvec3(tbMax[0], tbMax[1], tbMin[2]),
          glm::dvec3(tbMax[0], tbMax[1], tbMax[2]), glm::dvec3(tbMin[0], tbMax[1], tbMax[2])}};

  // Simple ray-sphere intersection test for every corner point.
  for (auto const& tbPnt : tbPnts) {
    double     dRayLength = glm::length(tbPnt - mCameraData.mCamPos);
    glm::dvec3 vRayDir    = (tbPnt - mCameraData.mCamPos) / dRayLength;
    double     b          = glm::dot(mCameraData.mCamPos, vRayDir);
    double     c          = glm::dot(mCameraData.mCamPos, mCameraData.mCamPos) -
               mHorizonCullRadius * mHorizonCullRadius;
    double fDet = b * b - c;

    // No intersection between corner and camera position: Tile visible!
    if (fDet < 0.0) {
      return true;
    }

    fDet = std::sqrt(fDet);

    // Both intersection points are behind the camera but tile is in front (presumes tiles to be
    // frustum culled already!!!). E.g. While travelling in a deep crater and looking above.
    if ((-b - fDet) < 0.0 && (-b + fDet) < 0.0) {
      return true;
    }

    // Tile in front of planet.
    if (dRayLength < -b - fDet) {
      return true;
    }
  }

  return false;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::lodbodies
