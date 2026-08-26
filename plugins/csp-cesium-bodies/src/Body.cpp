////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "Body.hpp"
#include "TilesetRenderer.hpp"
#include "logger.hpp"

#include "../../../../src/cs-core/SolarSystem.hpp"
#include "../../../../src/cs-scene/CelestialObserver.hpp"

#include <VistaKernel/DisplayManager/VistaDisplayManager.h>
#include <VistaKernel/DisplayManager/VistaProjection.h>
#include <VistaKernel/DisplayManager/VistaViewport.h>
#include <VistaKernel/VistaSystem.h>

#include <Cesium3DTilesSelection/Tileset.h>
#include <Cesium3DTilesSelection/TilesetExternals.h>
#include <Cesium3DTilesSelection/TilesetOptions.h>

namespace csp::cesiumbodies {

Body::Body(std::string const&                       name,
    Cesium3DTilesSelection::TilesetExternals const& tilesetExternals, int64_t assetId,
    std::string const& ionToken, Cesium3DTilesSelection::TilesetOptions const& options,
    std::shared_ptr<cs::core::SolarSystem> solarSystem,
    std::shared_ptr<cs::core::Settings>     settings)
    : mName(name) {
  mCelestialObject = solarSystem->getObject(name);

  auto bodyOptions      = options;
  bodyOptions.ellipsoid = CesiumGeospatial::Ellipsoid(mCelestialObject->getRadii().zxy());

  mTileset = std::make_unique<Cesium3DTilesSelection::Tileset>(
      tilesetExternals, assetId, ionToken, bodyOptions);

  mTilesetRenderer =
      std::make_shared<TilesetRenderer>(mTileset.get(), mCelestialObject, solarSystem, settings, mName);

  if (mCelestialObject) {
    mCelestialObject->setSurface(mTilesetRenderer);
    mCelestialObject->setIntersectableObject(mTilesetRenderer);
    logger().info("Registered as CelestialSurface for {}.", name);
  }

  logger().info(
      "Cesium Ion Tileset Created (Asset {}). Streaming will begin on first update.", assetId);
}

Body::~Body() {
  if (mCelestialObject) {
    mCelestialObject->setSurface(nullptr);
    mCelestialObject->setIntersectableObject(nullptr);
  }
}

void Body::update(cs::scene::CelestialObserver& observer) {
  if (!mCelestialObject)
    return;

  glm::dvec3 glmPos = observer.getPosition();
  glm::dvec3 camPositionECEF(glmPos.zxy());

  glm::dmat4 bodyToObserver = mCelestialObject->getObserverRelativeTransform();
  glm::dmat3 rot;
  if (double s = glm::length(glm::dvec3(bodyToObserver[0])); s > 0.0) {
    rot[0] = glm::dvec3(bodyToObserver[0]) / s;
    rot[1] = glm::dvec3(bodyToObserver[1]) / s;
    rot[2] = glm::dvec3(bodyToObserver[2]) / s;
  } else {
    rot = glm::dmat3(1.0);
  }
  glm::dvec3 glmDir = glm::normalize(glm::transpose(rot) * glm::dvec3(0.0, 0.0, -1.0));
  glm::dvec3 glmUp  = glm::normalize(glm::transpose(rot) * glm::dvec3(0.0, 1.0, 0.0));

  glm::dvec3 camDirectionECEF(glmDir.zxy());
  glm::dvec3 camUpECEF(glmUp.zxy());

  VistaViewport* pViewport = GetVistaSystem()->GetDisplayManager()->GetViewports().begin()->second;
  int            sizeX = 1920, sizeY = 1080; // fallback if query fails
  pViewport->GetViewportProperties()->GetSize(sizeX, sizeY);

  // CosmoScout's observer scale magnifies the view: Scale=0.2 means the world appears 5× bigger
  // (1/0.2 = 5). Each pixel therefore covers 1/5th the physical area. For Cesium's SSE formula
  // (SSE = geoError × viewportH / (dist × 2 × tan(vFov/2))), this magnification is equivalent
  // to having a proportionally larger viewport. This is exactly what csp-lod-bodies does
  // implicitly — its LODVisitor extracts the camera from the modelview matrix which has the
  // scale baked in (LODVisitor.cpp:62). We replicate this by scaling the viewport.
  // IMPORTANT: Only enlarge the viewport when Scale < 1.0 (magnified/close-up view).
  // At Scale >= 1.0 (orbit/far view), the physical viewport correctly represents the screen —
  // the camera IS physically far away and Cesium's SSE is naturally correct.
  // Without this clamp, orbital Scale=2.7M would produce a sub-pixel viewport → 0 LOD.
  double     scaleFactor = std::min(observer.getScale(), 1.0);
  glm::dvec2 viewportSize(
      static_cast<double>(sizeX) / scaleFactor, static_cast<double>(sizeY) / scaleFactor);

  double left = -0.5, right = 0.5, bottom = -0.5, top = 0.5;
  auto*  pProjProps = pViewport->GetProjection()->GetProjectionProperties();
  pProjProps->GetProjPlaneExtents(left, right, bottom, top);
  double hFov = 2.0 * std::atan((right - left) / 2.0);
  double vFov = 2.0 * std::atan((top - bottom) / 2.0);

  Cesium3DTilesSelection::ViewState viewState(
      camPositionECEF, camDirectionECEF, camUpECEF, viewportSize, hFov, vFov);

  std::vector frustums = {viewState};
  mTileset->updateViewGroup(mTileset->getDefaultViewGroup(), frustums);
  mTileset->loadTiles();
}

std::shared_ptr<cs::scene::CelestialSurface> Body::getSurface() const {
  return mTilesetRenderer;
}

} // namespace csp::cesiumbodies