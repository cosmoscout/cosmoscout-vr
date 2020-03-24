////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "CelestialAnchorNode.hpp"
#include "CelestialObserver.hpp"

#include <VistaKernel/GraphicsManager/VistaNodeBridge.h>

namespace cs::scene {

////////////////////////////////////////////////////////////////////////////////////////////////////

CelestialAnchorNode::CelestialAnchorNode(VistaGroupNode* pParent, IVistaNodeBridge* pBridge,
    std::string const& sName, std::string const& sCenterName, std::string const& sFrameName)
    : CelestialAnchor(sCenterName, sFrameName)
    , VistaTransformNode(pParent, pBridge, pBridge->NewTransformNodeData(), sName) {
  pParent->AddChild(this);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void CelestialAnchorNode::update(double tTime, CelestialObserver const& oObs) {
  try {
    glm::mat4 mat = oObs.getRelativeTransform(tTime, *this);

    SetTransform(VistaTransformMatrix(mat[0][0], mat[1][0], mat[2][0], mat[3][0], mat[0][1],
        mat[1][1], mat[2][1], mat[3][1], mat[0][2], mat[1][2], mat[2][2], mat[3][2], mat[0][3],
        mat[1][3], mat[2][3], mat[3][3]));
  } catch (...) {
    // data might be unavailable
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cs::scene
