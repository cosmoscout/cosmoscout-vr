////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "Trajectory.hpp"

#include "../../../src/cs-core/SolarSystem.hpp"
#include "../../../src/cs-utils/FrameTimings.hpp"
#include "logger.hpp"

#include <VistaKernel/GraphicsManager/VistaGraphicsManager.h>
#include <VistaKernel/GraphicsManager/VistaSceneGraph.h>
#include <VistaKernel/GraphicsManager/VistaTransformNode.h>
#include <VistaKernel/VistaSystem.h>
#include <VistaKernelOpenSGExt/VistaOpenSGMaterialTools.h>

namespace csp::trajectories {

////////////////////////////////////////////////////////////////////////////////////////////////////

Trajectory::Trajectory(std::shared_ptr<Plugin::Settings> pluginSettings,
    std::shared_ptr<cs::core::SolarSystem>               solarSystem)
    : mPluginSettings(std::move(pluginSettings))
    , mSolarSystem(std::move(solarSystem)) {

  pLength.connect([this](double val) {
    mPoints.clear();
    mTrajectory.setMaxAge(val * 24 * 60 * 60);
  });

  pColor.connect([this](glm::vec3 const& val) {
    mTrajectory.setStartColor(glm::vec4(val, 1.F));
    mTrajectory.setEndColor(glm::vec4(val, 0.F));
  });

  pSamples.connect([this](uint32_t /*value*/) { mPoints.clear(); });

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

void Trajectory::update(double tTime) {
  if (!mPluginSettings->mEnableTrajectories.get()) {
    return;
  }

  auto parent = mSolarSystem->getObject(mParentName);
  auto target = mSolarSystem->getObject(mTargetName);

  if (parent && target && parent->getIsInExistence() && target->getIsOrbitVisible()) {
    double dLengthSeconds = pLength.get() * 24.0 * 60.0 * 60.0;
    double dSampleLength  = dLengthSeconds / pSamples.get();

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

      auto startExistence = glm::max(parent->getExistence()[0], target->getExistence()[0]);
      auto endExistence   = glm::min(parent->getExistence()[1], target->getExistence()[1]);

      if (mLastUpdateTime < tTime) {
        if (completeRecalculation) {
          mLastSampleTime = tTime - dLengthSeconds - dSampleLength;
          mStartIndex     = 0;
        }

        while (mLastSampleTime < tTime) {
          mLastSampleTime += dSampleLength;

          try {
            double     tSampleTime = glm::clamp(mLastSampleTime, startExistence, endExistence);
            glm::dvec3 pos         = parent->getRelativePosition(tSampleTime, *target);
            mPoints[mStartIndex]   = glm::dvec4(pos.x, pos.y, pos.z, tSampleTime);

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
                glm::clamp(mLastSampleTime - dLengthSeconds, startExistence, endExistence);
            glm::dvec3 pos = parent->getRelativePosition(tSampleTime, *target);
            mPoints[(mStartIndex - 1 + pSamples.get()) % pSamples.get()] =
                glm::dvec4(pos.x, pos.y, pos.z, tSampleTime);

            mStartIndex = (mStartIndex - 1 + static_cast<int>(pSamples.get())) %
                          static_cast<int>(pSamples.get());
          } catch (...) {
            // Getting the relative transformation may fail due to insufficient SPICE data.
          }
        }
      }

      mLastUpdateTime = tTime;

      if (completeRecalculation) {
        logger().debug("Recalculating trajectory for {}.", mTargetName);
      }
    }

    mLastFrameTime = tTime;

    if (!mPoints.empty()) {
      glm::dvec3 tip = mPoints[mStartIndex];
      try {
        tip = parent->getRelativePosition(tTime, *target);
      } catch (...) {
        // Getting the relative transformation may fail due to insufficient SPICE data.
      }

      mTrajectory.upload(parent->getObserverRelativeTransform(), tTime, mPoints, tip, mStartIndex);
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Trajectory::setTargetName(std::string objectName) {
  mPoints.clear();
  mTargetName = std::move(objectName);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Trajectory::setParentName(std::string objectName) {
  mPoints.clear();
  mParentName = std::move(objectName);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::string const& Trajectory::getTargetName() const {
  return mTargetName;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::string const& Trajectory::getParentName() const {
  return mParentName;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool Trajectory::Do() {
  if (!mPluginSettings->mEnableTrajectories.get()) {
    return true;
  }

  auto parent = mSolarSystem->getObject(mParentName);
  auto target = mSolarSystem->getObject(mTargetName);

  if (parent->getIsInExistence() && target->getIsOrbitVisible()) {
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
