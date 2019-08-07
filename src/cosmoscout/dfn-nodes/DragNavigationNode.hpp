////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef VIRTUAL_PLANET_DRAG_NAVIGATION_NODE_HPP
#define VIRTUAL_PLANET_DRAG_NAVIGATION_NODE_HPP

#include <VistaKernel/VistaKernelConfig.h>

#include <VistaBase/VistaVectorMath.h>
#include <VistaDataFlowNet/VdfnNode.h>
#include <VistaDataFlowNet/VdfnNodeFactory.h>
#include <VistaDataFlowNet/VdfnPort.h>
#include <VistaDataFlowNet/VdfnSerializer.h>
#include <VistaKernel/GraphicsManager/VistaTransformNode.h>

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/gtx/transform.hpp>

#include <memory>

namespace cs::core {
class InputManager;
class TimeControl;
class SolarSystem;
} // namespace cs::core

/// This DFN-Node contains the logic for picking, dragging and rotating planets. It is used for the
/// mouse interaction as well as the flystick and HTC-Vive interaction.
/// You may have a look at config/vista/xml/dragging.xml for an example usage of this node.
class DragNavigationNode : public IVdfnNode {
 public:
  DragNavigationNode(std::shared_ptr<cs::core::SolarSystem> const& pSolarSystem,
      std::shared_ptr<cs::core::InputManager> const&               pInputManager,
      std::shared_ptr<cs::core::TimeControl> const&                pTimeControl);
  ~DragNavigationNode() override = default;

  bool PrepareEvaluationRun() override;

 protected:
  bool DoEvalNode() override;
  bool GetIsValid() const override;

 private:
  TVdfnPort<double>* mTime = nullptr;

  bool mStartInteractionInitialized = false;
  bool mDraggingPlanet              = false;
  bool mLocalRotation               = false;
  bool mDoRollCorrection            = false;

  glm::dvec3 mStartIntersection = glm::dvec3(0.0);
  glm::dvec3 mStartRayDir       = glm::dvec3(0.0);
  glm::dvec3 mStartObserverPos  = glm::dvec3(0.0);
  glm::dquat mStartObserverRot  = glm::dquat(1.0, 0.0, 0.0, 0.0);

  double     mTargetAngle      = 0.0;
  float      mCurrentAngleDiff = 0.f;
  glm::dvec3 mCurrentAxis      = glm::dvec3(1.0, 0.0, 0.0);

  VistaTransformNode* mSelectionTrans;

  std::shared_ptr<cs::core::SolarSystem>  mSolarSystem;
  std::shared_ptr<cs::core::InputManager> mInputManager;
  std::shared_ptr<cs::core::TimeControl>  mTimeControl;
};

class DragNavigationNodeCreate : public VdfnNodeFactory::IVdfnNodeCreator {
 public:
  DragNavigationNodeCreate(std::shared_ptr<cs::core::SolarSystem> const& pSolarSystem,
      std::shared_ptr<cs::core::InputManager> const&                     pInputManager,
      std::shared_ptr<cs::core::TimeControl> const&                      pTimeControl);

  IVdfnNode* CreateNode(const VistaPropertyList& oParams) const override;

 private:
  std::shared_ptr<cs::core::SolarSystem>  mSolarSystem;
  std::shared_ptr<cs::core::InputManager> mInputManager;
  std::shared_ptr<cs::core::TimeControl>  mTimeControl;
};

#endif // VIRTUAL_PLANET_DRAG_NAVIGATION_NODE_HPP
