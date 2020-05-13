////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "UpdateBoundsVisitor.hpp"

#include "PlanetParameters.hpp"
#include "RenderDataDEM.hpp"
#include "TreeManagerBase.hpp"

namespace csp::lodbodies {

////////////////////////////////////////////////////////////////////////////////////////////////////

/* explicit */
UpdateBoundsVisitor::UpdateBoundsVisitor(
    TreeManagerBase* treeMgrDEM, PlanetParameters const& params)
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
    auto*     rdDEM = mTreeMgrDEM->find<RenderDataDEM>(tileId);

    rdDEM->setBounds(calcTileBounds(
        *tile, mParams->mEquatorialRadius, mParams->mPolarRadius, mParams->mHeightScale));

    result = true;
  }

  return result;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool UpdateBoundsVisitor::preVisit(TileId const& tileId) {
  TileNode* node  = getState().mNodeDEM;
  TileBase* tile  = node->getTile();
  auto*     rdDEM = mTreeMgrDEM->find<RenderDataDEM>(tileId);

  rdDEM->setBounds(calcTileBounds(
      *tile, mParams->mEquatorialRadius, mParams->mPolarRadius, mParams->mHeightScale));

  return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::lodbodies
