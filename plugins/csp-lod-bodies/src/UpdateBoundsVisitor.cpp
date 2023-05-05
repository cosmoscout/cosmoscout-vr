////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "UpdateBoundsVisitor.hpp"

#include "PlanetParameters.hpp"
#include "TileBounds.hpp"
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
    auto* rdDEM = node->getTileData();

    rdDEM->setBounds(calcTileBounds(*rdDEM, mParams->mRadii, mParams->mHeightScale));

    result = true;
  }

  return result;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool UpdateBoundsVisitor::preVisit(TileId const& tileId) {
  TileNode* node  = getState().mNodeDEM;
  auto*     rdDEM = node->getTileData();

  rdDEM->setBounds(calcTileBounds(*rdDEM, mParams->mRadii, mParams->mHeightScale));

  return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::lodbodies
