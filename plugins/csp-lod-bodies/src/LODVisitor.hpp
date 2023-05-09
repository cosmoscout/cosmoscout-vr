////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSP_LOD_BODIES_LODVISITOR_HPP
#define CSP_LOD_BODIES_LODVISITOR_HPP

#include "Frustum.hpp"
#include "TileBounds.hpp"
#include "TileDataBase.hpp"
#include "TileId.hpp"
#include "TileVisitor.hpp"

#include <vector>

namespace csp::lodbodies {

struct PlanetParameters;
class TreeManager;

/// Specialization of TileVisitor that determines the necessary level of detail for tiles and
/// produces lists of tiles to load and draw respectively.
class LODVisitor : public TileVisitor<LODVisitor> {
 public:
  explicit LODVisitor(PlanetParameters const& params, TreeManager* treeMgr);

  void queueRecomputeTileBounds();

  int  getFrameCount() const;
  void setFrameCount(int frameCount);

  glm::ivec4 const& getViewport() const;
  void              setViewport(glm::ivec4 const& vp);

  glm::dmat4 const& getModelview() const;
  void              setModelview(glm::dmat4 const& m);

  glm::dmat4 const& getProjection() const;
  void              setProjection(glm::dmat4 const& m);

  /// Controls whether updates to the level of detail (LOD) decisions are made. When disabled
  /// previous decisions will be reused.
  ///
  /// This must have been enabled for at least one frame before it can be disabled, otherwise
  /// internal data is not correctly initialized!
  void setUpdateLOD(bool enable);
  bool getUpdateLOD() const;

  /// Controls whether updates to the culling decisions are mode. When disabled previous decisions
  /// will be reused.
  ///
  /// This must have been enabled for at least one frame before it can be disabled, otherwise
  /// internal data is not correctly initialized!
  void setUpdateCulling(bool enable);
  bool getUpdateCulling() const;

  /// Returns the nodes that should be loaded. The parent tiles of these have been
  /// determined to not provide sufficient resolution.
  std::vector<TileId> const& getLoadNodes() const;

  /// Returns the nodes that should be rendered.
  std::vector<TileNode*> const& getRenderNodes() const;

 private:
  /// Struct storing information relevant for LOD selection.
  struct LODData {
    glm::dmat4 mMatVM;
    glm::dmat4 mMatP;
    Frustum    mFrustumES; // frustum in eye space
    glm::ivec4 mViewport;
  };

  /// Struct storing information relevant for frustum culling.
  struct CullData {
    Frustum        mFrustumMS; // frustum in model space
    glm::f64mat3x3 mMatN;
    glm::dvec3     mCamPos;
  };

  bool preTraverse() override;
  void postTraverse() override;

  bool preVisitRoot(TileId const& tileId) override;
  bool preVisit(TileId const& tileId) override;

  void             pushState() override;
  void             popState() override;
  StateBase&       getState() override;
  StateBase const& getState() const override;

  /// Visit the node with given the tileId. Returns whether children should be visited.
  bool visitNode(TileId const& tileId);

  /// Handle the case where the node with given the tileId should be refined. Tests whether
  /// refinement is possible (i.e. whether data is loaded) and returns whether children should be
  /// visited.
  bool handleRefine(TileId const& tileId);

  void addLoadChildren(TileNode* node);

  /// Returns whether the currently visited node is potentially visible. Tests if the node's
  /// bounding box intersects the camera frustum.
  bool testVisible(TileId const& tileId);

  /// Returns whether the currently visited node should be refined, i.e. if it's children should be
  /// used to achieve desired resolution. Estimates the screen space size (in pixels) of the node
  /// and compares that with the desired LOD factor.
  bool testNeedRefine(TileId const& tileId);

  void drawLevel();

  friend class TileVisitor<LODVisitor>;

  static std::size_t const sMaxStackDepth = 32;

  PlanetParameters const* mParams;
  TreeManager*            mTreeMgr;
  bool                    mRecomputeTileBounds = false;

  glm::ivec4 mViewport;
  glm::dmat4 mMatVM;
  glm::dmat4 mMatP;
  LODData    mLodData;
  CullData   mCullData;

  std::vector<StateBase> mStack;
  int                    mStackTop;

  std::vector<TileId>    mLoadNodes;
  std::vector<TileNode*> mRenderNodes;

  int  mFrameCount;
  bool mUpdateLOD;
  bool mUpdateCulling;
};

} // namespace csp::lodbodies

#endif // CSP_LOD_BODIES_LODVISITOR_HPP
