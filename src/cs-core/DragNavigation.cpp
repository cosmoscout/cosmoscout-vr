////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "DragNavigation.hpp"

#include "logger.hpp"

#include "../cs-core/InputManager.hpp"
#include "../cs-core/SolarSystem.hpp"
#include "../cs-core/TimeControl.hpp"
#include "../cs-utils/convert.hpp"

#include <VistaDataFlowNet/VdfnObjectRegistry.h>
#include <VistaKernel/GraphicsManager/VistaSceneGraph.h>
#include <VistaKernel/VistaSystem.h>
#include <glm/gtx/io.hpp>
#include <utility>

namespace {

////////////////////////////////////////////////////////////////////////////////////////////////////

std::optional<double> IntersectSphere(glm::dvec3 const& origin, glm::dvec3 const& direction,
    glm::dvec3 const& center, double radius) {
  // Calculate the nearest hit point between ray and the rotation sphere
  double b    = glm::dot(-center + origin, direction);
  double c    = glm::dot(-center + origin, -center + origin) - (radius * radius);
  double fDet = b * b - c;

  // Coefficient of the nearest hit-point along the ray
  if (fDet > 0.0) {
    return -b - std::sqrt(fDet);
  }

  return {};
}

////////////////////////////////////////////////////////////////////////////////////////////////////

glm::dvec3 GetPositionInObserverFrame(cs::scene::CelestialAnchor const& anchor,
    std::shared_ptr<cs::core::SolarSystem> const&                       pSolarSystem,
    std::shared_ptr<cs::core::TimeControl> const&                       pTimeControl) {
  double                     simulationTime(pTimeControl->pSimulationTime.get());
  cs::scene::CelestialAnchor observerAnchor(
      pSolarSystem->getObserver().getCenterName(), pSolarSystem->getObserver().getFrameName());

  return observerAnchor.getRelativePosition(simulationTime, anchor);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace

namespace cs::core {

////////////////////////////////////////////////////////////////////////////////////////////////////

DragNavigation::DragNavigation(std::shared_ptr<cs::core::SolarSystem> pSolarSystem,
    std::shared_ptr<cs::core::InputManager>                           pInputManager,
    std::shared_ptr<cs::core::TimeControl>                            pTimeControl)
    : mSolarSystem(std::move(pSolarSystem))
    , mInputManager(std::move(pInputManager))
    , mTimeControl(std::move(pTimeControl))

{
  mSelectionTrans = dynamic_cast<VistaTransformNode*>(
      GetVistaSystem()->GetGraphicsManager()->GetSceneGraph()->GetNode("SELECTION_NODE"));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void DragNavigation::update() {
  // current observer transform of this frame
  glm::dvec3 observerPos = mSolarSystem->getObserver().getAnchorPosition();
  glm::dquat observerRot = mSolarSystem->getObserver().getAnchorRotation();

  // store observer transform when dragging started
  if (!mInputManager->pButtons[0].get() && !mInputManager->pButtons[1].get()) {
    mStartObserverPos = observerPos;
    mStartObserverRot = observerRot;
  }

  glm::dmat4 startObserverTransform(
      glm::translate(mStartObserverPos) *
      glm::rotate(glm::angle(mStartObserverRot), glm::axis(mStartObserverRot)));

  // get pick ray direction and origin
  VistaTransformMatrix tmp;
  mSelectionTrans->GetWorldTransform(tmp);
  glm::dmat4 pickTransform(tmp[0][0], tmp[1][0], tmp[2][0], tmp[3][0], tmp[0][1], tmp[1][1],
      tmp[2][1], tmp[3][1], tmp[0][2], tmp[1][2], tmp[2][2], tmp[3][2], tmp[0][3], tmp[1][3],
      tmp[2][3], tmp[3][3]);

  const glm::dvec3 rotationCenter(0.0, 0.0, 0.0);

  // The current ray transform in observer space is used to determine
  // the amount of rotation which is required
  glm::dvec3 rayDir = glm::normalize(
      (startObserverTransform * pickTransform * glm::dvec4(0.0, 0.0, -1.0, 0.0)).xyz());
  glm::dvec3 rayOrigin = pickTransform[3].xyz() * mSolarSystem->getObserver().getAnchorScale();
  rayOrigin            = (startObserverTransform * glm::vec4(rayOrigin, 1.0)).xyz();

  // store observer transform when dragging started
  if (!mInputManager->pButtons[0].get() && !mInputManager->pButtons[1].get()) {

    auto pickedPlanet = std::dynamic_pointer_cast<cs::scene::CelestialBody>(
        mInputManager->pHoveredObject.get().mObject);

    if (pickedPlanet) {
      // observer can be in another spice frame, therefore we need to
      // convert pick position to observer frame
      cs::scene::CelestialAnchor anchor(
          pickedPlanet->getCenterName(), pickedPlanet->getFrameName());
      anchor.setAnchorPosition(mInputManager->pHoveredObject.get().mPosition);

      try {
        mStartIntersection = GetPositionInObserverFrame(anchor, mSolarSystem, mTimeControl);
        mDraggingPlanet    = true;
      } catch (std::exception const& e) {
        // Getting the position in observer coordinates may fail due to insufficient SPICE data.
        logger().warn("Failed to grab '{}': {}", pickedPlanet->getCenterName(), e.what());
      }
    }

    float const epsilon = 0.05F;

    // if no planet is currently under the pointer or if we picked too close
    // too the horizon we will 'grab the sky' instead
    if (!pickedPlanet ||
        glm::dot(rayDir, glm::normalize(mStartIntersection - rotationCenter)) >= -epsilon) {
      mDraggingPlanet = false;
    }

    // when dragging in empty space, only the direction of the ray is required
    mStartRayDir = rayDir;

    mStartInteractionInitialized = true;
  }

  if (!mStartInteractionInitialized) {
    return;
  }

  float const smoothThreshold = 0.8F;

  if (mInputManager->pButtons[0].get() || mInputManager->pButtons[1].get()) {
    glm::dvec3 end_vec;
    glm::dvec3 start_vec;
    bool       bPerformRotation = false;
    mLocalRotation              = mInputManager->pButtons[1].get();

    if (mLocalRotation) {
      start_vec        = mStartRayDir;
      end_vec          = rayDir;
      bPerformRotation = true;
    } else if (mDraggingPlanet) {
      // The radius is used to rotate the camera around the target body on a sphere of this exact
      // size. This works well in most cases, however, if the target body is a rather extreme
      // ellipsoid, this actually changes the observer altitude which is not-so-nice. But for most
      // bodies in the Solar System this method works well.
      double radius = glm::length(mStartIntersection - rotationCenter);

      // Coefficient of the nearest hit-point along the ray
      auto t = IntersectSphere(rayOrigin, rayDir, rotationCenter, radius);

      // we do not want to drag something behind us
      if (t && *t > 0) {
        end_vec          = glm::normalize((rayOrigin + (*t * rayDir)) - rotationCenter);
        start_vec        = glm::normalize(mStartIntersection - rotationCenter);
        bPerformRotation = true;
      }
    } else {
      start_vec        = rayDir;
      end_vec          = mStartRayDir;
      bPerformRotation = true;
    }

    // Only rotate, if no other interactive object is affected
    if (bPerformRotation && !mInputManager->pActiveNode.get() &&
        !mInputManager->pActiveGuiItem.get()) {
      // Rotation angle computations:
      glm::dvec3 currentAxis = glm::cross(start_vec, end_vec);

      // Only if the vectors are not co-linear
      if (glm::length(currentAxis) > 0) {
        // The rotation axis is perpendicular to the start and the end position vectors
        mCurrentAxis = glm::normalize(currentAxis);

        // The final amount of camera rotation around camlanetCenter
        double dot         = glm::min(1.0, glm::dot(start_vec, end_vec));
        double targetAngle = -1.0 * std::acos(dot);

        // reduce rotation speed close to planet
        if (!mDraggingPlanet && !mLocalRotation && mSolarSystem->pActiveBody.get()) {
          double fac   = 0.5;
          auto   radii = mSolarSystem->pActiveBody.get()->getRadii();
          if (radii[0] > 0) {
            auto surfacePos = utils::convert::scaleToGeodeticSurface(observerPos, radii);
            auto distance   = observerPos - surfacePos;
            fac             = glm::length(distance) / glm::length(surfacePos);
          }

          // some magic numbers here to achieve 'good' dragging speeds
          // regardless of the surface distance when 'grabbing the sky'
          fac = glm::clamp(fac, 0.0001, 0.5);
          targetAngle *= fac * 4;
        }

        float const epsilon = 0.001F;
        // Prepare an animated rotation approaching target angle
        mDoKineticSmoothOut = std::abs(targetAngle - mTargetAngle) > epsilon;
        mCurrentAngleDiff += static_cast<float>(targetAngle - mTargetAngle);
        mTargetAngle = targetAngle;
      } else {
        mTargetAngle        = 0;
        mCurrentAngleDiff   = 0;
        mDoKineticSmoothOut = false;
      }
    } else {
      // Update camera for smoothing out even though button is pressed but
      // outside the ideal sphere
      mStartObserverPos = observerPos;
      mStartObserverRot = observerRot;

      // Prepare smoothing out remaining angle diff
      if (mCurrentAngleDiff != 0.F) {
        // Reduce inertia a little.
        mTargetAngle      = mCurrentAngleDiff * 0.5;
        mCurrentAngleDiff = 0.F;
      }

      // Smooth out remaining rotation do be done
      mTargetAngle        = smoothThreshold * mTargetAngle;
      mDoKineticSmoothOut = true;
    }
  } else {
    // Prepare smoothing out remaining angle diff
    if (mCurrentAngleDiff != 0.F) {
      if (mDoKineticSmoothOut) {
        // Reduce inertia a little.
        mTargetAngle = mCurrentAngleDiff * 0.5F;
      } else {
        mTargetAngle = 0.F;
      }
      mCurrentAngleDiff = 0.F;
    }

    // Smooth out remaining rotation do be done
    mTargetAngle = smoothThreshold * mTargetAngle;

    float const epsilon = 0.001F;
    if (std::abs(mTargetAngle) < epsilon) {
      mTargetAngle      = 0.F;
      mLocalRotation    = false;
      mDoRollCorrection = false;
    }
  }

  // Softly increasing the absolute rotation angle value
  mCurrentAngleDiff = smoothThreshold * mCurrentAngleDiff;

  // apply observer position change if rotating planet
  if (!mLocalRotation) {
    glm::dvec3 newObserverPos =
        (glm::translate(rotationCenter) * glm::rotate(mTargetAngle, mCurrentAxis) *
            glm::translate(-rotationCenter) * glm::dvec4(mStartObserverPos, 1.0))
            .xyz();
    mSolarSystem->getObserver().setAnchorPosition(newObserverPos);
  }

  glm::dquat newObserverRot = glm::angleAxis(mTargetAngle, mCurrentAxis) * mStartObserverRot;

  if (mLocalRotation && mSolarSystem->pActiveBody.get()) {
    // perform roll correction if observer is close to planet (10% of
    // radius) and planet normal is already close to up
    // or if the orthogonal on planet normal and up vector is
    // close to the viewers x axis (e.g. if the user is looking down or
    // upwards but the horizon is still straight)
    auto radii      = mSolarSystem->pActiveBody.get()->getRadii();
    auto surfacePos = utils::convert::scaleToGeodeticSurface(observerPos, radii);
    auto distance   = observerPos - surfacePos;

    glm::dvec3 normal = glm::normalize(distance);

    if (glm::length(observerPos) < glm::length(surfacePos) * 1.1) {
      glm::dvec3 y = glm::normalize((newObserverRot * glm::dvec4(0, 1, 0, 0)).xyz());
      glm::dvec3 x = glm::normalize((newObserverRot * glm::dvec4(1, 0, 0, 0)).xyz());

      double const epsilon = 0.999;
      if (glm::dot(normal, y) > epsilon ||
          (glm::dot(normal, y) > 0 &&
              std::abs(glm::dot(glm::normalize(glm::cross(y, normal)), x)) > epsilon)) {
        mDoRollCorrection = true;
      }
    }

    if (mDoRollCorrection) {
      glm::dvec3 z = (newObserverRot * glm::dvec4(0, 0, 1, 0)).xyz();
      glm::dvec3 x = -glm::cross(z, normal);
      glm::dvec3 y = -glm::cross(x, z);

      x = glm::normalize(x);
      y = glm::normalize(y);
      z = glm::normalize(z);

      newObserverRot = glm::toQuat(glm::dmat3(x, y, z));
    }
  }

  mSolarSystem->getObserver().setAnchorRotation(newObserverRot);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cs::core

////////////////////////////////////////////////////////////////////////////////////////////////////