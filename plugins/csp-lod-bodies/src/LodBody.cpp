////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "LodBody.hpp"

#include "../../../src/cs-core/GraphicsEngine.hpp"
#include "../../../src/cs-core/GuiManager.hpp"
#include "../../../src/cs-core/SolarSystem.hpp"
#include "../../../src/cs-gui/GuiItem.hpp"
#include "../../../src/cs-utils/FrameTimings.hpp"
#include "utils.hpp"

#include <VistaKernel/GraphicsManager/VistaGroupNode.h>
#include <VistaKernel/GraphicsManager/VistaOpenGLNode.h>
#include <VistaKernel/GraphicsManager/VistaSceneGraph.h>
#include <VistaKernel/VistaSystem.h>
#include <VistaKernelOpenSGExt/VistaOpenSGMaterialTools.h>

namespace csp::lodbodies {

////////////////////////////////////////////////////////////////////////////////////////////////////

LodBody::LodBody(std::shared_ptr<cs::core::Settings> const& settings,
    std::shared_ptr<cs::core::GraphicsEngine>               graphicsEngine,
    std::shared_ptr<cs::core::SolarSystem>                  solarSystem,
    std::shared_ptr<Plugin::Settings> const&                pluginSettings,
    std::shared_ptr<cs::core::GuiManager> const& pGuiManager, std::string const& sCenterName,
    std::string const& sFrameName, std::shared_ptr<GLResources> const& glResources,
    double tStartExistence, double tEndExistence)
    : cs::scene::CelestialBody(sCenterName, sFrameName, tStartExistence, tEndExistence)
    , mSettings(settings)
    , mGraphicsEngine(std::move(graphicsEngine))
    , mSolarSystem(std::move(solarSystem))
    , mPluginSettings(pluginSettings)
    , mGuiManager(pGuiManager)
    , mPlanet(glResources)
    , mShader(settings, pluginSettings, pGuiManager)
    , mRadii(cs::core::SolarSystem::getRadii(sCenterName)) {

  pVisible.connect([this](bool val) {
    if (val) {
      mGraphicsEngine->registerCaster(&mPlanet);
    } else {
      mGraphicsEngine->unregisterCaster(&mPlanet);
    }
  });

  mPlanet.setTerrainShader(&mShader);

  // per-planet settings -----------------------------------------------------
  mPlanet.setEquatorialRadius(static_cast<float>(mRadii[0]));
  mPlanet.setPolarRadius(static_cast<float>(mRadii[0]));
  pVisibleRadius = mRadii[0];

  // scene-wide settings -----------------------------------------------------
  mHeightScaleConnection = mSettings->mGraphics.pHeightScale.connectAndTouch(
      [this](float val) { mPlanet.setHeightScale(val); });

  mPluginSettings->mLODFactor.connectAndTouch([this](float val) { mPlanet.setLODFactor(val); });

  mPluginSettings->mEnableWireframe.connectAndTouch(
      [this](bool val) { mPlanet.getTileRenderer().setWireframe(val); });

  mPluginSettings->mEnableTilesFreeze.connectAndTouch([this](bool val) {
    mPlanet.getLODVisitor().setUpdateLOD(!val);
    mPlanet.getLODVisitor().setUpdateCulling(!val);
  });

  // Add to scenegraph.
  VistaSceneGraph* pSG = GetVistaSystem()->GetGraphicsManager()->GetSceneGraph();
  mGLNode.reset(pSG->NewOpenGLNode(pSG->GetRoot(), this));
  VistaOpenSGMaterialTools::SetSortKeyOnSubtree(
      mGLNode.get(), static_cast<int>(cs::utils::DrawOrder::ePlanets));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

LodBody::~LodBody() {
  mGraphicsEngine->unregisterCaster(&mPlanet);
  mSettings->mGraphics.pHeightScale.disconnect(mHeightScaleConnection);

  VistaSceneGraph* pSG = GetVistaSystem()->GetGraphicsManager()->GetSceneGraph();
  pSG->GetRoot()->DisconnectChild(mGLNode.get());
}

////////////////////////////////////////////////////////////////////////////////////////////////////

PlanetShader const& LodBody::getShader() const {
  return mShader;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void LodBody::setSun(std::shared_ptr<const cs::scene::CelestialObject> const& sun) {
  mSun = sun;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool LodBody::getIntersection(
    glm::dvec3 const& rayPos, glm::dvec3 const& rayDir, glm::dvec3& pos) const {
  return utils::intersectPlanet(&mPlanet, rayPos, rayDir, pos);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

double LodBody::getHeight(glm::dvec2 lngLat) const {
  return utils::getHeight(&mPlanet, HeightSamplePrecision::eActual, lngLat);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

glm::dvec3 LodBody::getRadii() const {
  return mRadii;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void LodBody::setDEMtileSource(std::shared_ptr<TileSource> source) {
  if (!source->isSame(mDEMtileSource.get())) {
    mPlanet.setDEMSource(source.get());
    mDEMtileSource = std::move(source);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void LodBody::setIMGtileSource(std::shared_ptr<TileSource> source) {
  if (source) {
    if (!source->isSame(mIMGtileSource.get())) {
      mPlanet.setIMGSource(source.get());
      mShader.pEnableTexture = true;
      mShader.pTextureIsRGB  = (source->getDataType() == TileDataType::eU8Vec3);
      mIMGtileSource         = std::move(source);
    }
  } else {
    if (mIMGtileSource) {
      mShader.pEnableTexture = false;
      mPlanet.setIMGSource(nullptr);
      mIMGtileSource = nullptr;
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::shared_ptr<TileSource> const& LodBody::getDEMtileSource() const {
  return mDEMtileSource;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::shared_ptr<TileSource> const& LodBody::getIMGtileSource() const {
  return mIMGtileSource;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void LodBody::update(double tTime, cs::scene::CelestialObserver const& oObs) {
  cs::scene::CelestialObject::update(tTime, oObs);

  if (getIsInExistence() && pVisible.get()) {
    mPlanet.setWorldTransform(getWorldTransform());

    if (mSun) {
      double sunIlluminance = 1.0;
      if (mSettings->mGraphics.pEnableHDR.get()) {
        sunIlluminance = mSolarSystem->getSunIlluminance(getWorldTransform()[3]);
      }

      auto sunDirection =
          glm::normalize(glm::inverse(getWorldTransform()) *
                         glm::dvec4(mSolarSystem->getSunDirection(getWorldTransform()[3]), 0.0));

      mShader.setSun(sunDirection, static_cast<float>(sunIlluminance));
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool LodBody::Do() {
  if (getIsInExistence() && pVisible.get()) {
    cs::utils::FrameTimings::ScopedTimer timer("LoD-Body " + getCenterName());
    mPlanet.Do();
  }

  return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool LodBody::GetBoundingBox(VistaBoundingBox& /*bb*/) {
  return false;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::lodbodies
