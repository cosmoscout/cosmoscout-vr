////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef VIRTUAL_PLANET_AUTO_SCENE_SCALE_NODE_HPP
#define VIRTUAL_PLANET_AUTO_SCENE_SCALE_NODE_HPP

#include <VistaBase/VistaVectorMath.h>
#include <VistaDataFlowNet/VdfnNode.h>
#include <VistaDataFlowNet/VdfnNodeFactory.h>
#include <VistaDataFlowNet/VdfnPort.h>
#include <VistaDataFlowNet/VdfnSerializer.h>
#include <VistaKernel/VistaKernelConfig.h>

#include <glm/glm.hpp>
#include <memory>

namespace cs::core {
class TimeControl;
class SolarSystem;
class GraphicsEngine;
} // namespace cs::core

/// DFN-node which scales the cs::scene::CelestialObserver of the given solar system to move the
/// closest body to a small world space distance. This distance depends on his or her *real*
/// distance in outer space to the respective body.
/// In order for the scientists to be able to interact with their environment, the next virtual
/// celestial body must never be more than an arm’s length away. If the Solar System were always
/// represented on a 1:1 scale, the virtual planetary surface would be too far away to work
/// effectively with the simulation.
/// As objects will be quite close to the observer in world space if the user is far away in *real*
/// space, this node also reduces the far clip distance in order to increase depth accuracy for
/// objects close to the observer.
/// This node also manages the SPICE frame changes when the observer moves from body to body.
/// You may have a look at config/vista/xml/auto_scenescale_desktop.xml for an example usage of this
/// node.
class AutoSceneScaleNode : public IVdfnNode {
 public:
  AutoSceneScaleNode(std::shared_ptr<cs::core::SolarSystem> const& pSolarSystem,
      std::shared_ptr<cs::core::GraphicsEngine> const&             graphicsEngine,
      std::shared_ptr<cs::core::TimeControl> const& pTimeControl, VistaPropertyList const& oParams);
  ~AutoSceneScaleNode() override = default;

  bool PrepareEvaluationRun() override;

 protected:
  bool DoEvalNode() override;

 private:
  std::shared_ptr<cs::core::SolarSystem>    mSolarSystem;
  std::shared_ptr<cs::core::GraphicsEngine> mGraphicsEngine;
  std::shared_ptr<cs::core::TimeControl>    mTimeControl;

  TVdfnPort<double>* mTime;

  // These values are set in the node's DFN configuration.

  /// The minimum observer scale denominator (default: 1)
  const double mMinScale;

  /// The minimum observer scale denominator (default: 10000000)
  const double mMaxScale;

  /// The near clip dista will always be set to this value (default: 0.2m)
  const double mNearClip;

  /// When the user is farther than mFarRealDistance away from the closest body, the far clip
  /// distance will be set to this value (default: 200m)
  const double mMinFarClip;

  /// When the user is closer than mCloseRealDistance away from the closest body, the far clip
  /// distance will be set to this value (default: 20000m)
  const double mMaxFarClip;

  /// When the user is closer than mCloseRealDistance away from the closest body, the observer will
  /// be scaled in such a way, that the closest body appears at this distance (default: 1.7m)
  const double mCloseVisualDistance;

  /// When the user is farther than mFarRealDistance away from the closest body, the observer will
  /// be scaled in such a way, that the closest body appears at this distance (default: 0.8m)
  const double mFarVisualDistance;

  /// Reference *real* world near-distance for the distances above (default: 1.7m)
  const double mCloseRealDistance;

  /// Reference *real* world near-distance for the distances above (default: 1000000m)
  const double mFarRealDistance;

  /// Every frame a weight is computed for every body. If the heighest weight exceeds this
  /// threshold, the observer will switch to the SPICE frame of that body.
  const double mLockWeight;

  /// Every frame a weight is computed for every body. If the heighest weight exceeds this
  /// threshold, the observer will switch to the SPICE center of that body.
  const double mTrackWeight;

  /// Very small objects are hard to fly-to. Therefore this size is assumed when objects are
  /// smaller. Default is 1000000m.
  const double mMinObjectSize;

  glm::dvec3 mLastObserverPosition;
  double     mLastTime;
};

class AutoSceneScaleNodeCreate : public VdfnNodeFactory::IVdfnNodeCreator {
 public:
  AutoSceneScaleNodeCreate(std::shared_ptr<cs::core::SolarSystem> const& pSolarSystem,
      std::shared_ptr<cs::core::GraphicsEngine> const&                   graphicsEngine,
      std::shared_ptr<cs::core::TimeControl> const&                      pTimeControl);
  IVdfnNode* CreateNode(VistaPropertyList const& oParams) const override;

 private:
  std::shared_ptr<cs::core::SolarSystem>    mSolarSystem;
  std::shared_ptr<cs::core::GraphicsEngine> mGraphicsEngine;
  std::shared_ptr<cs::core::TimeControl>    mTimeControl;
};

#endif // VIRTUAL_PLANET_AUTO_SCENE_SCALE_NODE_HPP
