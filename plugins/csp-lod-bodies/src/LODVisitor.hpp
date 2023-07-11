////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSP_LOD_BODIES_LODVISITOR_HPP
#define CSP_LOD_BODIES_LODVISITOR_HPP

#include "Frustum.hpp"
#include "TileId.hpp"
#include "TileVisitor.hpp"

#include <vector>

namespace csp::lodbodies {

struct PlanetParameters;
class TreeManager;

/// Specialization of TileVisitor that determines the necessary level of detail for tiles and
/// produces lists of tiles to load and draw respectively.
class LODVisitor : public TileVisitor {
 public:
  LODVisitor(PlanetParameters const& params, TreeManager* treeMgr);

  /// If called, node bounds will be recomputed during the next traversal. This should be called
  /// whenever the body radius or the elevation scale has been changed.
  void queueRecomputeTileBounds();

  /// Used to compute the age of unused tiles.
  void setFrameCount(int frameCount);

  /// These are required for frustum culling and level-of-detail selection.
  void setModelview(glm::dmat4 const& m);
  void setProjection(glm::dmat4 const& m);

  /// Controls whether the tree cut should be modified. When disabled previous decisions will be
  /// reused.
  ///
  /// This must have been enabled for at least one frame before it can be disabled, otherwise
  /// internal data is not correctly initialized!
  void setUpdateLOD(bool enable);
  bool getUpdateLOD() const;

  /// Returns the nodes that should be loaded. The parent tiles of these have been
  /// determined to not provide sufficient resolution.
  std::vector<TileId> const& getLoadNodes() const;

  /// Returns the nodes that should be rendered.
  std::vector<TileNode*> const& getRenderNodes() const;

 private:
  /// Struct storing camera information.
  struct CameraData {
    Frustum        mFrustumES; // frustum in eye space
    Frustum        mFrustumMS; // frustum in model space
    glm::f64mat3x3 mMatN;
    glm::dvec3     mCamPos;
  };

  bool preTraverse() override;
  void postTraverse() override;

  bool preVisitRoot(TileNode* root) override;
  bool preVisit(TileNode* node) override;

  /// Visit the given node. Returns whether children should be visited.
  bool visitNode(TileNode* node);

  /// Returns whether the currently visited node should be refined, i.e. if it's children should be
  /// used to achieve desired resolution. Estimates the screen space size (in pixels) of the node
  /// and compares that with the desired LOD factor.
  bool testNeedRefine(TileNode* node) const;

  // Returns if the tile bounds intersect the current frustum. For each plane of the frustum
  // determine if any corner of the bounding box is inside the plane's halfspace. If all corners are
  // outside one halfspace the bounding box is outside the frustum and the algorithm stops early.
  //
  // TODO There is potential for optimization here, the paper "Optimized View Frustum Culling -
  // Algorithms for Bounding Boxes" http://www.cse.chalmers.se/~uffe/vfc_bbox.pdf contains ideas
  // (for example how to avoid testing all 8 corners).
  bool testInFrustum(TileNode* node) const;

  // Returns true if one the eight tile bbox corner points is not occluded by a proxy sphere.
  // Culls tiles behind the horizon.
  bool testFrontFacing(TileNode* node) const;

  PlanetParameters const* mParams;
  TreeManager*            mTreeMgr;
  bool                    mRecomputeTileBounds = false;

  glm::dmat4 mMatVM;
  glm::dmat4 mMatP;
  CameraData mCameraData;
  double     mHorizonCullRadius;

  std::vector<TileId>    mLoadNodes;
  std::vector<TileNode*> mRenderNodes;

  int  mFrameCount;
  bool mUpdateLOD;
};

} // namespace csp::lodbodies

#endif // CSP_LOD_BODIES_LODVISITOR_HPP
