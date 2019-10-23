////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "ObserverNavigationNode.hpp"

#include "../cs-core/InputManager.hpp"
#include "../cs-core/SolarSystem.hpp"

#include <VistaAspects/VistaPropertyAwareable.h>

////////////////////////////////////////////////////////////////////////////////////////////////////

ObserverNavigationNode::ObserverNavigationNode(
    std::shared_ptr<cs::core::SolarSystem> const&  pSolarSystem,
    std::shared_ptr<cs::core::InputManager> const& pInputManager, VistaPropertyList const& oParams)
    : IVdfnNode()
    , mSolarSystem(pSolarSystem)
    , mInputManager(pInputManager)
    , mTime(nullptr)
    , mTranslation(nullptr)
    , mRotation(nullptr)
    , mOffset(nullptr)
    , mPreventNavigationWhenHoveredGui(
          oParams.GetValueOrDefault<bool>("prevent_navigation_when_hovered_gui", true))
    , mMaxAngularSpeed(oParams.GetValueOrDefault<double>("max_angular_speed", glm::pi<double>()))
    , mMaxLinearSpeed(oParams.GetValueOrDefault<VistaVector3D>(
          "max_linear_speed", VistaVector3D(1.0, 1.0, 1.0)))
    , mAngularDirection(1.0, 0.0, 0.0, 0.0)
    , mAngularSpeed(0.0)
    , mAngularDeceleration(oParams.GetValueOrDefault<double>("angular_deceleration", 0.1))
    , mLinearDirection(0.0)
    , mLinearSpeed(0.0)
    , mLinearDeceleration(oParams.GetValueOrDefault<double>("linear_deceleration", 0.1))
    , mLastTime(-1.0) {
  mLinearSpeed.mDirection = cs::utils::AnimationDirection::eLinear;

  RegisterInPortPrototype("time", new TVdfnPortTypeCompare<TVdfnPort<double>>);
  RegisterInPortPrototype("translation", new TVdfnPortTypeCompare<TVdfnPort<VistaVector3D>>);
  RegisterInPortPrototype("rotation", new TVdfnPortTypeCompare<TVdfnPort<VistaQuaternion>>);
  RegisterInPortPrototype("offset", new TVdfnPortTypeCompare<TVdfnPort<VistaVector3D>>);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool ObserverNavigationNode::PrepareEvaluationRun() {
  mTime        = dynamic_cast<TVdfnPort<double>*>(GetInPort("time"));
  mTranslation = dynamic_cast<TVdfnPort<VistaVector3D>*>(GetInPort("translation"));
  mRotation    = dynamic_cast<TVdfnPort<VistaQuaternion>*>(GetInPort("rotation"));
  mOffset      = dynamic_cast<TVdfnPort<VistaVector3D>*>(GetInPort("offset"));
  return GetIsValid();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool ObserverNavigationNode::GetIsValid() const {
  return ((mTranslation || mRotation || mOffset) && mTime);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool ObserverNavigationNode::DoEvalNode() {
  double dTtime = mTime->GetValue();

  // skip first update since we do not know the delta time
  if (mLastTime < 0.0) {
    mLastTime = dTtime;
    return true;
  }

  // ensures that we don't loose time on many very small updates
  double dDeltaTime = dTtime - mLastTime;
  if (dDeltaTime < Vista::Epsilon) {
    return true;
  }

  mLastTime = dTtime;

  if (mTranslation) {
    auto tmp              = mTranslation->GetValue();
    auto vLinearDirection = glm::dvec3(tmp[0], tmp[1], tmp[2]);

    // update velocities
    if (glm::length(vLinearDirection) > 0.0) {
      if (!mInputManager->pHoveredGuiNode.get() || !mPreventNavigationWhenHoveredGui) {
        mLinearDirection = vLinearDirection;

        if (mLinearSpeed.mEndValue == 0.0) {
          mLinearSpeed.mStartValue = 1.0;
          mLinearSpeed.mEndValue   = 1.0;
        }
      }
    } else {
      if (mLinearSpeed.mEndValue == 1.0) {
        mLinearSpeed.mStartValue = mLinearSpeed.get(dTtime);
        mLinearSpeed.mEndValue   = 0.0;
        mLinearSpeed.mStartTime  = dTtime;
        mLinearSpeed.mEndTime    = dTtime + mLinearDeceleration;
      }
    }
  }

  if (mRotation) {
    auto          tmp       = mRotation->GetValue().GetNormalized().GetAxisAndAngle();
    VistaVector3D axis      = tmp.m_v3Axis;
    double        angle     = tmp.m_fAngle;
    auto          qRotation = glm::angleAxis(angle, glm::dvec3(axis[0], axis[1], axis[2]));

    // update velocities
    if (angle > 0.0) {
      if (!mInputManager->pHoveredGuiNode.get() || !mPreventNavigationWhenHoveredGui) {
        mAngularDirection = qRotation;

        if (mAngularSpeed.mEndValue == 0.0) {
          mAngularSpeed.mStartValue = 1.0;
          mAngularSpeed.mEndValue   = 1.0;
        }
      }
    } else {
      if (mAngularSpeed.mEndValue == 1.0) {
        mAngularSpeed.mStartValue = mAngularSpeed.get(dTtime);
        mAngularSpeed.mEndValue   = 0.0;
        mAngularSpeed.mStartTime  = dTtime;
        mAngularSpeed.mEndTime    = dTtime + mAngularDeceleration;
      }
    }
  }

  glm::dvec3 vOffset(0.0);

  if (mOffset) {
    auto tmp = mOffset->GetValue();
    vOffset  = glm::dvec3(tmp[0], tmp[1], tmp[2]);
  }

  auto&      oObs              = mSolarSystem->getObserver();
  glm::dvec3 vObserverPosition = oObs.getAnchorPosition();
  glm::dquat qObserverRotation = oObs.getAnchorRotation();
  double     dObserverScale    = oObs.getAnchorScale();

  auto vTranslation = mLinearDirection;

  vTranslation.x *= mMaxLinearSpeed[0];
  vTranslation.y *= mMaxLinearSpeed[1];
  vTranslation.z *= mMaxLinearSpeed[2];
  vTranslation *= dObserverScale;
  vTranslation *= mLinearSpeed.get(dTtime);

  vTranslation *= dDeltaTime;
  vTranslation += vOffset * dObserverScale;

  auto       qRotation     = mAngularDirection;
  glm::dvec3 vRotationAxis = glm::axis(qRotation);
  double     dRotationAngle =
      glm::angle(qRotation) * dDeltaTime * mMaxAngularSpeed * mAngularSpeed.get(dTtime);

  if (glm::length(vTranslation) > 0.0) {
    oObs.setAnchorPosition(vObserverPosition + qObserverRotation * vTranslation);
  }

  if (dRotationAngle != 0.0) {
    oObs.setAnchorRotation(qObserverRotation * glm::angleAxis(dRotationAngle, vRotationAxis));
  }

  return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

ObserverNavigationNodeCreate::ObserverNavigationNodeCreate(
    std::shared_ptr<cs::core::SolarSystem> const&  pSolarSystem,
    std::shared_ptr<cs::core::InputManager> const& pInputManager)
    : mSolarSystem(pSolarSystem)
    , mInputManager(pInputManager) {
}

////////////////////////////////////////////////////////////////////////////////////////////////////

IVdfnNode* ObserverNavigationNodeCreate::CreateNode(const VistaPropertyList& oParams) const {
  return new ObserverNavigationNode(
      mSolarSystem, mInputManager, oParams.GetSubListConstRef("param"));
}

////////////////////////////////////////////////////////////////////////////////////////////////////
