////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CS_CORE_DRAG_NAVIGATION_HPP
#define CS_CORE_DRAG_NAVIGATION_HPP

#include "cs_core_export.hpp"

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

/// This class contains the logic for picking, dragging and rotating planets. It is used for the
/// mouse interaction as well as the flystick and HTC-Vive interaction.
class CS_CORE_EXPORT DragNavigation {
 public:
  DragNavigation(std::shared_ptr<cs::core::SolarSystem> pSolarSystem,
      std::shared_ptr<cs::core::InputManager>           pInputManager,
      std::shared_ptr<cs::core::TimeControl>            pTimeControl);

  DragNavigation(DragNavigation const& other) = delete;
  DragNavigation(DragNavigation&& other)      = delete;

  DragNavigation& operator=(DragNavigation const& other) = delete;
  DragNavigation& operator=(DragNavigation&& other) = delete;

  ~DragNavigation() = default;

  void update();

 private:
  std::shared_ptr<cs::core::SolarSystem>  mSolarSystem;
  std::shared_ptr<cs::core::InputManager> mInputManager;
  std::shared_ptr<cs::core::TimeControl>  mTimeControl;

  VistaTransformNode* mSelectionTrans;

  bool       mStartInteractionInitialized = false;
  bool       mDraggingPlanet              = false;
  bool       mLocalRotation               = false;
  bool       mDoRollCorrection            = false;
  bool       mDoKineticSmoothOut          = false;
  double     mTargetAngle                 = 0.0;
  float      mCurrentAngleDiff            = 0.F;
  glm::dvec3 mCurrentAxis                 = glm::dvec3(1.0, 0.0, 0.0);

  glm::dvec3 mStartIntersection = glm::dvec3(0.0);
  glm::dvec3 mStartRayDir       = glm::dvec3(0.0);
  glm::dvec3 mStartObserverPos  = glm::dvec3(0.0);
  glm::dquat mStartObserverRot  = glm::dquat(1.0, 0.0, 0.0, 0.0);
};

} // namespace cs::core

#endif // CS_CORE_DRAG_NAVIGATION_HPP
