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

Trajectory::Trajectory(
    std::shared_ptr<Plugin::Settings> pluginSettings, std::shared_ptr<cs::core::Settings> settings)
    : mPluginSettings(std::move(pluginSettings))
    , mSettings(std::move(settings)) {

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

  if (mPluginSettings->mEnableTrajectories.get() && getIsInExistence()) {
    double dSampleLength = dLengthSeconds / pSamples.get();

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
            double     tSampleTime = glm::clamp(mLastSampleTime, mExistence[0], mExistence[1]);
            glm::dvec3 pos         = getRelativePosition(tSampleTime, mTarget);
            mPoints[mStartIndex]   = glm::dvec4(pos.x, pos.y, pos.z, tSampleTime);

            setRadii(glm::max(glm::dvec3(glm::length(pos)), getRadii()));

            mStartIndex = (mStartIndex + 1) % static_cast<int>(pSamples.get());
          } catch (...) {
            // Getting the relative transformation may fail due to insufficient SPICE data.
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
                glm::clamp(mLastSampleTime - dLengthSeconds, mExistence[0], mExistence[1]);
            glm::dvec3 pos = getRelativePosition(tSampleTime, mTarget);
            mPoints[(mStartIndex - 1 + pSamples.get()) % pSamples.get()] =
                glm::dvec4(pos.x, pos.y, pos.z, tSampleTime);

            mStartIndex = (mStartIndex - 1 + static_cast<int>(pSamples.get())) %
                          static_cast<int>(pSamples.get());
            setRadii(glm::max(glm::dvec3(glm::length(pos)), getRadii()));
          } catch (...) {
            // Getting the relative transformation may fail due to insufficient SPICE data.
          }
        }
      }

      mLastUpdateTime = tTime;

      if (completeRecalculation) {
        logger().debug("Recalculating trajectory for {}.", mTargetAnchorName);
      }
    }

    mLastFrameTime = tTime;

    if (pVisible.get() && !mPoints.empty()) {
      glm::dvec3 tip = mPoints[mStartIndex];
      try {
        tip = getRelativePosition(tTime, mTarget);
      } catch (...) {
        // Getting the relative transformation may fail due to insufficient SPICE data.
      }

      mTrajectory.upload(matWorldTransform, tTime, mPoints, tip, mStartIndex);
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Trajectory::setTargetAnchorName(std::string const& anchorName) {
  mPoints.clear();
  mTargetAnchorName = anchorName;
  mSettings->initAnchor(mTarget, anchorName);
  updateExistence();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Trajectory::setParentAnchorName(std::string const& anchorName) {
  mPoints.clear();
  mParentAnchorName = anchorName;
  mSettings->initAnchor(*this, anchorName);
  updateExistence();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::string const& Trajectory::getTargetAnchorName() const {
  return mTargetAnchorName;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::string const& Trajectory::getParentAnchorName() const {
  return mParentAnchorName;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool Trajectory::Do() {
  if (mPluginSettings->mEnableTrajectories.get() && pVisible.get() && getIsInExistence()) {
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

void Trajectory::updateExistence() {
  if (!mTargetAnchorName.empty() && !mParentAnchorName.empty()) {
    auto parentExistence = mSettings->getAnchorExistence(mParentAnchorName);
    auto targetExistence = mSettings->getAnchorExistence(mTargetAnchorName);

    setExistence(glm::dvec2(std::max(parentExistence[0], targetExistence[0]),
        std::min(parentExistence[1], targetExistence[1])));
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::trajectories
