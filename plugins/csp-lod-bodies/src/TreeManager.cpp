////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "TreeManager.hpp"

#include "PlanetParameters.hpp"

namespace csp::lodbodies {

////////////////////////////////////////////////////////////////////////////////////////////////////

template <>
/* virtual */ RenderData* TreeManager<RenderDataDEM>::allocateRenderData(TileNode* node) {
  RenderDataDEM* rdata = mPool.construct();

  // init rdata
  rdata->setNode(node);
  rdata->setLastFrame(0);

  rdata->setBounds(calcTileBounds(
      *node->getTile(), mParams->mEquatorialRadius, mParams->mPolarRadius, mParams->mHeightScale));

  return rdata;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::lodbodies
