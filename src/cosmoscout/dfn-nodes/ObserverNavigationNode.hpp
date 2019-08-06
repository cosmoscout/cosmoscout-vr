////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef VIRTUAL_PLANET_OBSERVER_NODE_HPP
#define VIRTUAL_PLANET_OBSERVER_NODE_HPP

#include <VistaBase/VistaVectorMath.h>
#include <VistaDataFlowNet/VdfnNode.h>
#include <VistaDataFlowNet/VdfnNodeFactory.h>
#include <VistaDataFlowNet/VdfnPort.h>
#include <VistaDataFlowNet/VdfnSerializer.h>
#include <VistaKernel/VistaKernelConfig.h>

#include "../../cs-utils/AnimatedValue.hpp"

#include <memory>

namespace cs::core {
class InputManager;
class SolarSystem;
} // namespace cs::core

/// Node to apply navigation to an cs::scene::CelestialObserver.
/// The node has four inports - "translation", "rotation, "offset" and "time". The only mandatory
/// inport is "time". Any value fed into "translation" or "rotation" will be multiplied by "time"
/// and applied each frame to the observer. "offset" is considered to be an absolute change in
/// position and will not be multiplied with "time".
/// This node is used in many DFN-networks, you may have a look at config/vista/xml/mouse_zoom.xml
/// or config/vista/xml/headtracking.xml for example usages of this node.
class ObserverNavigationNode : public IVdfnNode {
 public:
  ObserverNavigationNode(std::shared_ptr<cs::core::SolarSystem> const& pSolarSystem,
      std::shared_ptr<cs::core::InputManager> const&                   pInputManager,
      VistaPropertyList const&                                         oParams);
  ~ObserverNavigationNode() override = default;

  bool PrepareEvaluationRun() override;

 protected:
  bool DoEvalNode() override;
  bool GetIsValid() const override;

 private:
  std::shared_ptr<cs::core::SolarSystem>  mSolarSystem;
  std::shared_ptr<cs::core::InputManager> mInputManager;

  TVdfnPort<double>*          mTime;
  TVdfnPort<VistaVector3D>*   mTranslation;
  TVdfnPort<VistaQuaternion>* mRotation;
  TVdfnPort<VistaVector3D>*   mOffset;

  const bool          mPreventNavigationWhenHoveredGui;
  const double        mMaxAngularSpeed;
  const VistaVector3D mMaxLinearSpeed;

  glm::dquat                       mAngularDirection;
  cs::utils::AnimatedValue<double> mAngularSpeed;
  const double                     mAngularDeceleration;

  glm::dvec3                       mLinearDirection;
  cs::utils::AnimatedValue<double> mLinearSpeed;
  const double                     mLinearDeceleration;

  double mLastTime;
};

class ObserverNavigationNodeCreate : public VdfnNodeFactory::IVdfnNodeCreator {
 public:
  ObserverNavigationNodeCreate(std::shared_ptr<cs::core::SolarSystem> const& pSolarSystem,
      std::shared_ptr<cs::core::InputManager> const&                         pInputManager);
  IVdfnNode* CreateNode(const VistaPropertyList& oParams) const override;

 private:
  std::shared_ptr<cs::core::SolarSystem>  mSolarSystem;
  std::shared_ptr<cs::core::InputManager> mInputManager;
};

#endif // VIRTUAL_PLANET_OBSERVER_NODE_HPP
