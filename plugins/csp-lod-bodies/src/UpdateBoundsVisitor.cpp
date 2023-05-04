////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "UpdateBoundsVisitor.hpp"

#include "PlanetParameters.hpp"
#include "RenderData.hpp"
#include "TreeManager.hpp"

namespace csp::lodbodies {

////////////////////////////////////////////////////////////////////////////////////////////////////

/* explicit */
UpdateBoundsVisitor::UpdateBoundsVisitor(TreeManager* treeMgrDEM, PlanetParameters const& params)
    : TileVisitor<UpdateBoundsVisitor>(treeMgrDEM->getTree(), nullptr)
    , mTreeMgrDEM(treeMgrDEM)
    , mParams(&params) {
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool UpdateBoundsVisitor::preTraverse() {
  return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool UpdateBoundsVisitor::preVisitRoot(TileId const& tileId) {
  bool      result = false;
  TileNode* node   = getState().mNodeDEM;

  if (node) {
    TileBase* tile  = node->getTile();
    auto*     rdDEM = mTreeMgrDEM->find(tileId);

    rdDEM->setBounds(calcTileBounds(*tile, mParams->mRadii, mParams->mHeightScale));

    result = true;
  }

  return result;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool UpdateBoundsVisitor::preVisit(TileId const& tileId) {
  TileNode* node  = getState().mNodeDEM;
  TileBase* tile  = node->getTile();
  auto*     rdDEM = mTreeMgrDEM->find(tileId);

  rdDEM->setBounds(calcTileBounds(*tile, mParams->mRadii, mParams->mHeightScale));

  return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::lodbodies
