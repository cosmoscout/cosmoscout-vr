////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "AutoSceneScaleNode.hpp"

#include "../../cs-core/GraphicsEngine.hpp"
#include "../../cs-core/SolarSystem.hpp"
#include "../../cs-core/TimeControl.hpp"
#include "../../cs-utils/convert.hpp"

#include <VistaAspects/VistaPropertyAwareable.h>
#include <VistaKernel/DisplayManager/VistaDisplayManager.h>
#include <VistaKernel/DisplayManager/VistaProjection.h>
#include <VistaKernel/VistaSystem.h>

////////////////////////////////////////////////////////////////////////////////////////////////////

AutoSceneScaleNode::AutoSceneScaleNode(std::shared_ptr<cs::core::SolarSystem> const& pSolarSystem,
    std::shared_ptr<cs::core::GraphicsEngine> const&                                 graphicsEngine,
    std::shared_ptr<cs::core::TimeControl> const& pTimeControl, VistaPropertyList const& oParams)
    : IVdfnNode()
    , mSolarSystem(pSolarSystem)
    , mGraphicsEngine(graphicsEngine)
    , mTimeControl(pTimeControl)
    , mTime(nullptr)
    , mMinScale(oParams.GetValueOrDefault<double>("min_scale", 1.0))
    , mMaxScale(oParams.GetValueOrDefault<double>("max_scale", 10000000.0))
    , mNearClip(oParams.GetValueOrDefault<double>("near_clip", 0.2))
    , mMinFarClip(oParams.GetValueOrDefault<double>("min_far_clip", 200.0))
    , mMaxFarClip(oParams.GetValueOrDefault<double>("max_far_clip", 20000.0))
    , mCloseVisualDistance(oParams.GetValueOrDefault<double>("close_visual_distance", 1.7))
    , mFarVisualDistance(oParams.GetValueOrDefault<double>("far_visual_distance", 0.8))
    , mCloseRealDistance(oParams.GetValueOrDefault<double>("close_real_distance", 1.7))
    , mFarRealDistance(oParams.GetValueOrDefault<double>("far_real_distance", 1000000.0))
    , mLockWeight(oParams.GetValueOrDefault<double>("lock_weight", 0.1))
    , mTrackWeight(oParams.GetValueOrDefault<double>("track_weight", 0.001))
    , mMinObjectSize(oParams.GetValueOrDefault<double>("min_object_size", 1000000.0))
    , mLastObserverPosition(0.0)
    , mLastTime(-1.0) {
  RegisterInPortPrototype("time", new TVdfnPortTypeCompare<TVdfnPort<double>>);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool AutoSceneScaleNode::PrepareEvaluationRun() {
  mTime = dynamic_cast<TVdfnPort<double>*>(GetInPort("time"));
  return GetIsValid();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool AutoSceneScaleNode::DoEvalNode() {
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

  auto&  oObs           = mSolarSystem->getObserver();
  double simulationTime = mTimeControl->pSimulationTime.get();

  // user will be locked to active planet, scene will be scaled that closest planet
  // is mScaleDistance away in world space
  std::shared_ptr<cs::scene::CelestialBody> pClosestBody;
  std::shared_ptr<cs::scene::CelestialBody> pActiveBody;

  double dActiveWeight    = 0;
  double dClosestDistance = std::numeric_limits<double>::max();

  glm::dvec3 vClosestPlanetObserverPosition(0.0);

  for (auto const& object : mSolarSystem->getBodies()) {
    if (!object->getIsInExistence()) {
      continue;
    }

    auto radii = object->getRadii();

    if (radii.x <= 0.0 || radii.y <= 0.0 || radii.z <= 0.0) {
      continue;
    }

    glm::dvec3 vObserverPos;

    try {
      vObserverPos = object->getRelativePosition(simulationTime, oObs);
    } catch (...) { continue; }

    double dDistance = glm::length(vObserverPos) - radii[0];
    double dWeight   = (radii[0] + mMinObjectSize) /
                     std::max(radii[0] + mMinObjectSize, radii[0] + dDistance - mMinObjectSize);

    if (dWeight > dActiveWeight) {
      pActiveBody   = object;
      dActiveWeight = dWeight;
    }

    if (dDistance < dClosestDistance) {
      pClosestBody                   = object;
      dClosestDistance               = dDistance;
      vClosestPlanetObserverPosition = vObserverPos;
    }
  }

  // change frame and center if there is a object with weight larger than mLockWeight
  // and mTrackWeight
  if (pActiveBody) {
    if (!oObs.isAnimationInProgress()) {
      std::string sCenter = "Solar System Barycenter";
      std::string sFrame  = "J2000";

      if (dActiveWeight > mLockWeight) {
        sFrame = pActiveBody->getFrameName();
      }

      if (dActiveWeight > mTrackWeight) {
        sCenter = pActiveBody->getCenterName();
      }

      mSolarSystem->pActiveBody     = pActiveBody;
      mSolarSystem->pObserverCenter = sCenter;
      mSolarSystem->pObserverFrame  = sFrame;
    }
  }

  // scale scene in such a way that the closest planet
  // is mScaleDistance away in world space
  if (pClosestBody) {
    auto   dSurfaceHeight = 0.0;
    double dRealDistance  = glm::length(vClosestPlanetObserverPosition);

    auto radii = pClosestBody->getRadii();

    if (radii[0] > 0) {
      auto lngLatHeight =
          cs::utils::convert::toLngLatHeight(vClosestPlanetObserverPosition, radii[0], radii[0]);
      dRealDistance = lngLatHeight.z;
      dRealDistance -=
          pClosestBody->getHeight(lngLatHeight.xy()) * mGraphicsEngine->pHeightScale.get();
    }

    if (std::isnan(dRealDistance)) {
      return true;
    }

    double interpolate = 1.0;

    if (mFarRealDistance != mCloseRealDistance) {
      interpolate = glm::clamp(
          (dRealDistance - mCloseRealDistance) / (mFarRealDistance - mCloseRealDistance), 0.0, 1.0);
    }

    double dScale = dRealDistance / glm::mix(mCloseVisualDistance, mFarVisualDistance, interpolate);
    dScale        = glm::clamp(dScale, mMinScale, mMaxScale);
    oObs.setAnchorScale(dScale);

    if (dRealDistance < mCloseRealDistance) {
      double     penetration = mCloseRealDistance - dRealDistance;
      glm::dvec3 position    = oObs.getAnchorPosition();
      oObs.setAnchorPosition(position + glm::normalize(position) * penetration);
    }

    // set far clip dynamically
    auto projections = GetVistaSystem()->GetDisplayManager()->GetProjections();
    for (auto const& projection : projections) {
      projection.second->GetProjectionProperties()->SetClippingRange(
          mNearClip, glm::mix(mMaxFarClip, mMinFarClip, interpolate));
    }
  }

  // update speed display
  mSolarSystem->pCurrentObserverSpeed =
      glm::length(mLastObserverPosition - oObs.getAnchorPosition()) / dDeltaTime;
  mLastObserverPosition = oObs.getAnchorPosition();

  return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

AutoSceneScaleNodeCreate::AutoSceneScaleNodeCreate(
    std::shared_ptr<cs::core::SolarSystem> const&    pSolarSystem,
    std::shared_ptr<cs::core::GraphicsEngine> const& graphicsEngine,
    std::shared_ptr<cs::core::TimeControl> const&    pTimeControl)
    : mSolarSystem(pSolarSystem)
    , mGraphicsEngine(graphicsEngine)
    , mTimeControl(pTimeControl) {
}

////////////////////////////////////////////////////////////////////////////////////////////////////

IVdfnNode* AutoSceneScaleNodeCreate::CreateNode(const VistaPropertyList& oParams) const {
  return new AutoSceneScaleNode(
      mSolarSystem, mGraphicsEngine, mTimeControl, oParams.GetSubListConstRef("param"));
}

////////////////////////////////////////////////////////////////////////////////////////////////////
