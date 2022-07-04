////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CS_SCENE_CELESTIAL_ANCHOR_NODE_HPP
#define CS_SCENE_CELESTIAL_ANCHOR_NODE_HPP

#include "CelestialAnchor.hpp"

#include <VistaKernel/GraphicsManager/VistaTransformNode.h>

namespace cs::scene {

class CelestialObserver;

/// A CelestialAnchorNode is a CelestialAnchor which is placed into the VistaSceneGraph.
class CS_SCENE_EXPORT CelestialAnchorNode : public CelestialAnchor, public VistaTransformNode {
 public:
  /// @param pParent The node to which this node will be added as a child.
  CelestialAnchorNode(VistaGroupNode* pParent, IVistaNodeBridge* pBridge,
      std::string const& sName = "", std::string const& sCenterName = "Solar System Barycenter",
      std::string const& sFrameName = "J2000");

  /// Updates the VistaTransformMatrix.
  void update(double tTime, CelestialObserver const& oObs);
};

} // namespace cs::scene

#endif // CS_SCENE_CELESTIAL_ANCHOR_NODE_HPP
