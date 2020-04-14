////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "ObserverNavigationNode.hpp"

#include "../cs-core/SolarSystem.hpp"
#include "../cs-gui/GuiItem.hpp"

#include <VistaAspects/VistaPropertyAwareable.h>

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

  auto&  oObs     = mSolarSystem->getObserver();
  double stepSize = glm::length(vTranslation);

  if (stepSize > 0.0) {
    // Ensure that an SolarSystem::updateSceneScale() is called at least at 100 Hz. If it is called
    // only once a frame, it can happen that the observer instantly travels to a planet's surface.
    auto steps = static_cast<int32_t>(std::ceil(dDeltaTime * 100.0));

    for (int32_t i(1); i <= steps; ++i) {
      oObs.setAnchorPosition(oObs.getAnchorPosition() + oObs.getAnchorRotation() * vTranslation *
                                                            oObs.getAnchorScale() /
                                                            static_cast<double>(steps));
      if (i < steps) {
        mSolarSystem->updateSceneScale();
      }
    }
  }

  auto       qRotation     = mAngularDirection;
  glm::dvec3 vRotationAxis = glm::axis(qRotation);
  double     dRotationAngle =
      glm::angle(qRotation) * dDeltaTime * mMaxAngularSpeed * mAngularSpeed.get(dTtime);

  if (dRotationAngle != 0.0) {
    oObs.setAnchorRotation(
        oObs.getAnchorRotation() * glm::angleAxis(dRotationAngle, vRotationAxis));
  }

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
