////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "TreeManager.hpp"

#include "PlanetParameters.hpp"

namespace csp::lodbodies {

////////////////////////////////////////////////////////////////////////////////////////////////////

template <>
/* virtual */ RenderData* TreeManager<RenderData>::allocateRenderData(TileNode* node) {
  RenderData* rdata = mPool.construct();

  // init rdata
  rdata->setNode(node);
  rdata->setLastFrame(0);

  rdata->setBounds(calcTileBounds(*node->getTile(), mParams->mRadii, mParams->mHeightScale));

  return rdata;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::lodbodies
