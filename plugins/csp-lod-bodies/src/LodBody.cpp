////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

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

LodBody::LodBody(std::shared_ptr<cs::core::Settings> settings,
    std::shared_ptr<cs::core::GraphicsEngine>        graphicsEngine,
    std::shared_ptr<cs::core::SolarSystem>           solarSystem,
    std::shared_ptr<Plugin::Settings>                pluginSettings,
    std::shared_ptr<cs::core::GuiManager> pGuiManager, std::shared_ptr<GLResources> glResources)
    : mSettings(std::move(settings))
    , mGraphicsEngine(std::move(graphicsEngine))
    , mSolarSystem(std::move(solarSystem))
    , mPluginSettings(std::move(pluginSettings))
    , mGuiManager(std::move(pGuiManager))
    , mEclipseShadowReceiver(
          std::make_shared<cs::core::EclipseShadowReceiver>(mSettings, mSolarSystem, false))
    , mPlanet(std::move(glResources), mPluginSettings->mTileResolutionDEM.get())
    , mShader(mSettings, mPluginSettings, mGuiManager, mEclipseShadowReceiver) {

  mGraphicsEngine->registerCaster(&mPlanet);
  mPlanet.setTerrainShader(&mShader);

  // scene-wide settings -----------------------------------------------------
  mHeightScaleConnection = mSettings->mGraphics.pHeightScale.connectAndTouch(
      [this](float val) { mPlanet.setHeightScale(val); });

  mPluginSettings->mLODFactor.connectAndTouch([this](float val) { mPlanet.setLODFactor(val); });

  mPluginSettings->mEnableWireframe.connectAndTouch(
      [this](bool val) { mPlanet.getTileRenderer().setWireframe(val); });

  mPluginSettings->mEnableBounds.connectAndTouch(
      [this](bool val) { mPlanet.getTileRenderer().setDrawBounds(val); });

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

bool LodBody::getIntersection(
    glm::dvec3 const& rayPos, glm::dvec3 const& rayDir, glm::dvec3& pos) const {
  return utils::intersectPlanet(&mPlanet, rayPos, rayDir, pos);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

double LodBody::getHeight(glm::dvec2 lngLat) const {
  return utils::getHeight(&mPlanet, HeightSamplePrecision::eActual, lngLat);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void LodBody::setDEMtileSource(std::shared_ptr<TileSource> source, uint32_t maxLevel) {
  if (!source->isSame(mDEMtileSource.get())) {
    mPlanet.setDEMSource(source.get());
    mDEMtileSource = std::move(source);
  }

  mMaxLevelDEM = maxLevel;

  mPlanet.setMaxLevel(std::max(mMaxLevelIMG, mMaxLevelDEM));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void LodBody::setIMGtileSource(std::shared_ptr<TileSource> source, uint32_t maxLevel) {
  if (source) {
    if (!source->isSame(mIMGtileSource.get())) {
      mPlanet.setIMGSource(source.get());
      mShader.pEnableTexture = true;
      mIMGtileSource         = std::move(source);
    }

    mMaxLevelIMG = maxLevel;
  } else {
    mShader.pEnableTexture = false;
    if (mIMGtileSource) {
      mPlanet.setIMGSource(nullptr);
      mIMGtileSource = nullptr;
    }

    mMaxLevelIMG = 0;
  }

  mPlanet.setMaxLevel(std::max(mMaxLevelIMG, mMaxLevelDEM));
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

void LodBody::setObjectName(std::string objectName) {
  mShader.setObjectName(objectName);
  mObjectName = std::move(objectName);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::string const& LodBody::getObjectName() const {
  return mObjectName;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void LodBody::update() {
  auto parent  = mSolarSystem->getObject(mObjectName);
  bool visible = parent && parent->getIsBodyVisible();

  mPlanet.setEnabled(visible);

  if (visible) {

    auto const& transform = parent->getObserverRelativeTransform();

    mPlanet.setRadii(parent->getRadii());
    mPlanet.setWorldTransform(parent->getObserverRelativeTransform());

    double sunIlluminance = mSolarSystem->getSunIlluminance(transform[3]);

    auto sunDirection = glm::normalize(
        glm::inverse(transform) * glm::dvec4(mSolarSystem->getSunDirection(transform[3]), 0.0));

    mShader.setSun(sunDirection, static_cast<float>(sunIlluminance));

    mEclipseShadowReceiver->update(*parent);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool LodBody::Do() {
  cs::utils::FrameTimings::ScopedTimer timer("LoD-Body " + mObjectName);
  mPlanet.draw();

  return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool LodBody::GetBoundingBox(VistaBoundingBox& /*bb*/) {
  return false;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::lodbodies
