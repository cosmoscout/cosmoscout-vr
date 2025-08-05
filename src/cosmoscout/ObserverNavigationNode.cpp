////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "ObserverNavigationNode.hpp"

#include "../cs-core/SolarSystem.hpp"
#include "../cs-gui/GuiItem.hpp"
#include "../cs-utils/convert.hpp"

#include <VistaAspects/VistaPropertyAwareable.h>
#include <glm/gtx/io.hpp>

////////////////////////////////////////////////////////////////////////////////////////////////////

ObserverNavigationNode::ObserverNavigationNode(
    cs::core::SolarSystem* pSolarSystem, VistaPropertyList const& oParams)
    : mSolarSystem(pSolarSystem)
    , mTime(nullptr)
    , mTranslation(nullptr)
    , mRotation(nullptr)
    , mOffset(nullptr)
    , mPreventNavigationWhenHoveredGui(
          oParams.GetValueOrDefault<bool>("prevent_navigation_when_hovered_gui", true))
    , mFixedHorizon(oParams.GetValueOrDefault<bool>("fixed_horizon", false))
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

  // NOLINTNEXTLINE(cppcoreguidelines-owning-memory): deleted in IVdfnNode::~IVdfnNode()
  RegisterInPortPrototype("time", new TVdfnPortTypeCompare<TVdfnPort<double>>);

  // NOLINTNEXTLINE(cppcoreguidelines-owning-memory): deleted in IVdfnNode::~IVdfnNode()
  RegisterInPortPrototype("translation", new TVdfnPortTypeCompare<TVdfnPort<VistaVector3D>>);

  // NOLINTNEXTLINE(cppcoreguidelines-owning-memory): deleted in IVdfnNode::~IVdfnNode()
  RegisterInPortPrototype("rotation", new TVdfnPortTypeCompare<TVdfnPort<VistaQuaternion>>);

  // NOLINTNEXTLINE(cppcoreguidelines-owning-memory): deleted in IVdfnNode::~IVdfnNode()
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
      mLinearDirection = vLinearDirection;

      if (mLinearSpeed.mEndValue == 0.0) {
        mLinearSpeed.mStartValue = 1.0;
        mLinearSpeed.mEndValue   = 1.0;
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
      mAngularDirection = qRotation;

      if (mAngularSpeed.mEndValue == 0.0) {
        mAngularSpeed.mStartValue = 1.0;
        mAngularSpeed.mEndValue   = 1.0;
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

  auto vTranslation = mLinearDirection;

  vTranslation.x *= mMaxLinearSpeed[0];
  vTranslation.y *= mMaxLinearSpeed[1];
  vTranslation.z *= mMaxLinearSpeed[2];
  vTranslation *= mLinearSpeed.get(dTtime);

  vTranslation *= dDeltaTime;
  vTranslation += vOffset;

  auto& oObs        = mSolarSystem->getObserver();
  auto  newPosition = oObs.getPosition() + oObs.getRotation() * vTranslation * oObs.getScale();
  oObs.setPosition(newPosition);

  auto       qRotation     = mAngularDirection;
  glm::dvec3 vRotationAxis = glm::axis(qRotation);
  double     dRotationAngle =
      glm::angle(qRotation) * dDeltaTime * mMaxAngularSpeed * mAngularSpeed.get(dTtime);

  auto newRotation =
      glm::normalize(oObs.getRotation() * glm::angleAxis(dRotationAngle, vRotationAxis));

  // If mFixedHorizon is set, we rotate the observer so that the horizon of the active object is
  // always leveled. For now, this breaks if we are in outer space or looking straight up or down.
  // But it can be very useful in cases were we know that the user is always close to a planet.
  if (mFixedHorizon && mSolarSystem->pActiveObject.get()) {
    auto radii      = mSolarSystem->pActiveObject.get()->getRadii();
    auto surfacePos = cs::utils::convert::scaleToGeodeticSurface(newPosition, radii);
    auto distance   = newPosition - surfacePos;

    glm::dvec3 normal = glm::normalize(distance);

    glm::dvec3 z = (newRotation * glm::dvec4(0, 0, 1, 0)).xyz();
    glm::dvec3 x = -glm::cross(z, normal);
    glm::dvec3 y = -glm::cross(x, z);

    x = glm::normalize(x);
    y = glm::normalize(y);
    z = glm::normalize(z);

    newRotation = glm::toQuat(glm::dmat3(x, y, z));
  }

  oObs.setRotation(newRotation);

  return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

ObserverNavigationNodeCreate::ObserverNavigationNodeCreate(cs::core::SolarSystem* pSolarSystem)
    : mSolarSystem(pSolarSystem) {
}

////////////////////////////////////////////////////////////////////////////////////////////////////

IVdfnNode* ObserverNavigationNodeCreate::CreateNode(const VistaPropertyList& oParams) const {
  return new ObserverNavigationNode(mSolarSystem, oParams.GetSubListConstRef("param"));
}

////////////////////////////////////////////////////////////////////////////////////////////////////
