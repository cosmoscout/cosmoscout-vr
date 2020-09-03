////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "Trajectory.hpp"

#include "../../../src/cs-scene/CelestialObserver.hpp"
#include "../../../src/cs-utils/FrameTimings.hpp"
#include "logger.hpp"

#include <VistaKernel/GraphicsManager/VistaGraphicsManager.h>
#include <VistaKernel/GraphicsManager/VistaSceneGraph.h>
#include <VistaKernel/GraphicsManager/VistaTransformNode.h>
#include <VistaKernel/VistaSystem.h>
#include <VistaKernelOpenSGExt/VistaOpenSGMaterialTools.h>

namespace csp::trajectories {

////////////////////////////////////////////////////////////////////////////////////////////////////

Trajectory::Trajectory(std::shared_ptr<Plugin::Settings> pluginSettings, std::string sTargetCenter,
    std::string sTargetFrame, std::string const& sParentCenter, std::string const& sParentFrame,
    double tStartExistence, double tEndExistence)
    : cs::scene::CelestialObject(sParentCenter, sParentFrame, tStartExistence, tEndExistence)
    , mPluginSettings(std::move(pluginSettings))
    , mTargetCenter(std::move(sTargetCenter))
    , mTargetFrame(std::move(sTargetFrame))
    , mStartIndex(0)
    , mLastUpdateTime(-1.0) {

  pLength.connect([this](double val) {
    mPoints.clear();
    mTrajectory.setMaxAge(val * 24 * 60 * 60);
  });

  pColor.connect([this](glm::vec3 const& val) {
    mTrajectory.setStartColor(glm::vec4(val, 1.F));
    mTrajectory.setEndColor(glm::vec4(val, 0.F));
  });

  pSamples.connect([this](uint32_t /*value*/) { mPoints.clear(); });

  mTrajectory.setUseLinearDepthBuffer(true);

  // Add to scenegraph.
  VistaSceneGraph* pSG = GetVistaSystem()->GetGraphicsManager()->GetSceneGraph();
  mGLNode.reset(pSG->NewOpenGLNode(pSG->GetRoot(), this));
  VistaOpenSGMaterialTools::SetSortKeyOnSubtree(
      mGLNode.get(), static_cast<int>(cs::utils::DrawOrder::eTransparentItems) - 1);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

Trajectory::~Trajectory() {
  VistaSceneGraph* pSG = GetVistaSystem()->GetGraphicsManager()->GetSceneGraph();
  pSG->GetRoot()->DisconnectChild(mGLNode.get());
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Trajectory::update(double tTime, cs::scene::CelestialObserver const& oObs) {
  cs::scene::CelestialObject::update(tTime, oObs);

  double dLengthSeconds = pLength.get() * 24.0 * 60.0 * 60.0;
  mTrailIsInExistence   = (tTime > mStartExistence && tTime < mEndExistence + dLengthSeconds);

  if (mPluginSettings->mEnableTrajectories.get() && mTrailIsInExistence) {
    double dSampleLength = dLengthSeconds / pSamples.get();

    cs::scene::CelestialAnchor target(mTargetCenter, mTargetFrame);

    // only recalculate if there is not too much change from frame to frame
    if (std::abs(mLastFrameTime - tTime) <= dLengthSeconds / 10.0) {
      // make sure to re-sample entire trajectory if complete reset is required
      bool completeRecalculation = false;

      if (mPoints.size() != pSamples.get()) {
        mPoints.resize(pSamples.get());
        completeRecalculation = true;
      }

      if (tTime > mLastSampleTime + dLengthSeconds || tTime < mLastSampleTime - dLengthSeconds) {
        completeRecalculation = true;
      }

      if (mLastUpdateTime < tTime) {
        if (completeRecalculation) {
          mLastSampleTime = tTime - dLengthSeconds - dSampleLength;
          mStartIndex     = 0;
        }

        while (mLastSampleTime < tTime) {
          mLastSampleTime += dSampleLength;

          try {
            double     tSampleTime = glm::clamp(mLastSampleTime, mStartExistence, mEndExistence);
            glm::dvec3 pos         = getRelativePosition(tSampleTime, target);
            mPoints[mStartIndex]   = glm::dvec4(pos.x, pos.y, pos.z, tSampleTime);

            pVisibleRadius = std::max(glm::length(pos), pVisibleRadius.get());

            mStartIndex = (mStartIndex + 1) % static_cast<int>(pSamples.get());
          } catch (...) {
            // data might be unavailable
          }
        }
      } else {
        if (completeRecalculation) {
          mLastSampleTime = tTime + dLengthSeconds + dSampleLength;
          mStartIndex     = 0;
        }

        while (mLastSampleTime - dSampleLength > tTime) {
          mLastSampleTime -= dSampleLength;

          try {
            double tSampleTime =
                glm::clamp(mLastSampleTime - dLengthSeconds, mStartExistence, mEndExistence);
            glm::dvec3 pos = getRelativePosition(tSampleTime, target);
            mPoints[(mStartIndex - 1 + pSamples.get()) % pSamples.get()] =
                glm::dvec4(pos.x, pos.y, pos.z, tSampleTime);

            mStartIndex = (mStartIndex - 1 + static_cast<int>(pSamples.get())) %
                          static_cast<int>(pSamples.get());
            pVisibleRadius = std::max(glm::length(pos), pVisibleRadius.get());
          } catch (...) {
            // data might be unavailable
          }
        }
      }

      mLastUpdateTime = tTime;

      if (completeRecalculation) {
        logger().debug("Recalculating trajectory for {}.", mTargetCenter);
      }
    }

    mLastFrameTime = tTime;

    if (pVisible.get()) {
      glm::dvec3 tip = getRelativePosition(tTime, target);
      mTrajectory.upload(matWorldTransform, tTime, mPoints, tip, mStartIndex);
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Trajectory::setTargetCenterName(std::string const& sCenterName) {
  if (mTargetCenter != sCenterName) {
    mPoints.clear();
    mTargetCenter = sCenterName;
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Trajectory::setTargetFrameName(std::string const& sFrameName) {
  if (mTargetFrame != sFrameName) {
    mPoints.clear();
    mTargetFrame = sFrameName;
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::string const& Trajectory::getTargetCenterName() const {
  return mTargetCenter;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::string const& Trajectory::getTargetFrameName() const {
  return mTargetFrame;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Trajectory::setCenterName(std::string const& sCenterName) {
  if (sCenterName != getCenterName()) {
    mPoints.clear();
  }
  cs::scene::CelestialObject::setCenterName(sCenterName);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Trajectory::setFrameName(std::string const& sFrameName) {
  if (sFrameName != getFrameName()) {
    mPoints.clear();
  }
  cs::scene::CelestialObject::setFrameName(sFrameName);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool Trajectory::Do() {
  if (mPluginSettings->mEnableTrajectories.get() && pVisible.get() && mTrailIsInExistence) {
    cs::utils::FrameTimings::ScopedTimer timer("Trajectories");
    mTrajectory.Do();
  }

  return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool Trajectory::GetBoundingBox(VistaBoundingBox& /*bb*/) {
  return false;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::trajectories
