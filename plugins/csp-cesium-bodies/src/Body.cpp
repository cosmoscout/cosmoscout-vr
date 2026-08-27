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

#include <Cesium3DTilesSelection/BoundingVolume.h>
#include <Cesium3DTilesSelection/Tileset.h>
#include <Cesium3DTilesSelection/TilesetExternals.h>
#include <Cesium3DTilesSelection/TilesetOptions.h>
#include <CesiumGeospatial/Cartographic.h>
#include <CesiumGeospatial/Ellipsoid.h>

#include <glm/gtx/norm.hpp>

#include <Cesium3DTilesSelection/ITilesetHeightSampler.h>
#include <CesiumGeometry/IntersectionTests.h>
#include <CesiumGeometry/Ray.h>
#include <CesiumGltfContent/GltfUtilities.h>
#include <limits>

namespace CesiumGeometry {
class Ray;
}
namespace csp::cesiumbodies {

////////////////////////////////////////////////////////////////////////////////////////////////////

Body::Body(std::string const&                       name,
    Cesium3DTilesSelection::TilesetExternals const& tilesetExternals, int64_t assetId,
    std::string const& ionToken, Cesium3DTilesSelection::TilesetOptions const& options,
    std::shared_ptr<cs::core::SolarSystem> solarSystem,
    std::shared_ptr<cs::core::Settings>    settings)
    : mName(name)
    , mSolarSystem(std::move(solarSystem))
    , mCelestialObject(mSolarSystem->getObject(name)) {

  auto bodyOptions      = options;
  bodyOptions.ellipsoid = CesiumGeospatial::Ellipsoid(mCelestialObject->getRadii().zxy());

  mTileset = std::make_unique<Cesium3DTilesSelection::Tileset>(
      tilesetExternals, assetId, ionToken, bodyOptions);

  mTilesetRenderer = std::make_shared<TilesetRenderer>(
      mTileset.get(), mCelestialObject, mSolarSystem, settings, mName);

  logger().info(
      "Cesium Ion Tileset Created (Asset {}). Streaming will begin on first update.", assetId);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

Body::~Body() = default;

////////////////////////////////////////////////////////////////////////////////////////////////////

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

////////////////////////////////////////////////////////////////////////////////////////////////////

static bool boundingVolumeContainsCoordinate(
    const Cesium3DTilesSelection::BoundingVolume& boundingVolume,

    const CesiumGeometry::Ray& ray, const CesiumGeospatial::Cartographic& coordinate,
    const CesiumGeospatial::Ellipsoid& ellipsoid) {

  struct Operation {
    const CesiumGeometry::Ray&            ray;
    const CesiumGeospatial::Cartographic& coordinate;
    const CesiumGeospatial::Ellipsoid&    ellipsoid;

    bool operator()(const CesiumGeometry::OrientedBoundingBox& boundingBox) const noexcept {
      std::optional<double> t =
          CesiumGeometry::IntersectionTests::rayOBBParametric(ray, boundingBox);
      return t && t.value() >= 0;
    }

    bool operator()(const CesiumGeospatial::BoundingRegion& boundingRegion) const noexcept {
      return boundingRegion.getRectangle().contains(coordinate);
    }

    bool operator()(const CesiumGeometry::BoundingSphere& boundingSphere) const noexcept {
      std::optional<double> t =
          CesiumGeometry::IntersectionTests::raySphereParametric(ray, boundingSphere);
      return t && t.value() >= 0;
    }

    bool operator()(const CesiumGeospatial::BoundingRegionWithLooseFittingHeights& boundingRegion)
        const noexcept {
      return boundingRegion.getBoundingRegion().getRectangle().contains(coordinate);
    }

    bool operator()(const CesiumGeospatial::S2CellBoundingVolume& s2Cell) const noexcept {
      return s2Cell.computeBoundingRegion(ellipsoid).getRectangle().contains(coordinate);
    }

    bool operator()(const CesiumGeometry::BoundingCylinderRegion& cylinderRegion) const noexcept {
      std::optional<double> t = CesiumGeometry::IntersectionTests::rayOBBParametric(
          ray, cylinderRegion.toOrientedBoundingBox());
      return t && t.value() >= 0;
    }
  };

  return std::visit(
      Operation{.ray = ray, .coordinate = coordinate, .ellipsoid = ellipsoid}, boundingVolume);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

double Body::getHeight(glm::dvec2 lngLat) const {
  // The ray for height queries starts at this fraction of the ellipsoid max
  // radius above the ellipsoid surface. If a tileset surface is more than this
  // distance above the ellipsoid, it may be missed by height queries.
  // 0.007 is chosen to accomodate Olympus Mons, the tallest peak on Mars. 0.007
  // is seven-tenths of a percent, or about 44,647 meters for WGS84, well above
  // the highest point on Earth.
  constexpr double rayOriginHeightFraction = 0.007;

  auto ellipsoid = mTileset->getEllipsoid();

  CesiumGeospatial::Cartographic position(lngLat.x, lngLat.y, 0.0);
  CesiumGeospatial::Cartographic start(position.longitude, position.latitude,
      ellipsoid.getMaximumRadius() * rayOriginHeightFraction);
  CesiumGeometry::Ray            ray(
      ellipsoid.cartographicToCartesian(start), -ellipsoid.geodeticSurfaceNormal(start));

  std::vector<Cesium3DTilesSelection::Tile::Pointer> candidateTiles{};
  std::vector<Cesium3DTilesSelection::Tile::Pointer> additiveCandidateTiles{};

  std::function<void(Cesium3DTilesSelection::Tile::Pointer const&)> findCandidateTiles =
      [&](Cesium3DTilesSelection::Tile::Pointer const& tile) -> void {
    // If tile failed to load, this means we can't complete the intersection
    if (tile->getState() == Cesium3DTilesSelection::TileLoadState::Failed) {
      return;
    }

    const std::optional<Cesium3DTilesSelection::BoundingVolume>& boundingVolume =
        tile->getContentBoundingVolume();

    if (tile->getChildren().empty()) { // This is a leaf node, it's a candidate
      if (boundingVolume) { // If optional content bounding volume exists, test against it
        if (boundingVolumeContainsCoordinate(*boundingVolume, ray, position, ellipsoid)) {
          candidateTiles.emplace_back(tile);
        }
      } else {
        candidateTiles.emplace_back(tile);
      }
    } else { // We have children
      // If additive refinement, add parent to the list with children
      if (tile->getRefine() == Cesium3DTilesSelection::TileRefine::Add) {
        if (boundingVolume) { // If optional content bounding volume exists, test against it
          if (boundingVolumeContainsCoordinate(*boundingVolume, ray, position, ellipsoid)) {
            additiveCandidateTiles.emplace_back(tile);
          }
        } else {
          additiveCandidateTiles.emplace_back(tile);
        }
      }

      for (Cesium3DTilesSelection::Tile& child : tile->getChildren()) {
        if (!boundingVolumeContainsCoordinate(child.getBoundingVolume(), ray, position, ellipsoid))
          continue;

        findCandidateTiles(&child);
      }
    }
  };

  findCandidateTiles(mTileset->getRootTile());

  std::optional<CesiumGltfContent::GltfUtilities::RayGltfHit> intersection;
  auto intersectVisibleTile = [&](Cesium3DTilesSelection::Tile* tile) {
    Cesium3DTilesSelection::TileRenderContent* renderContent =
        tile->getContent().getRenderContent();
    if (!renderContent)
      return;

    auto [hit, _] = CesiumGltfContent::GltfUtilities::intersectRayGltfModel(
        ray, renderContent->getModel(), true, tile->getTransform());

    // Set ray info to this hit if closer, or the first hit
    if (!intersection.has_value()) {
      intersection = std::move(hit);
    } else if (hit) {
      double prevDistSq = intersection->rayToWorldPointDistanceSq;
      if (double thisDistSq = hit->rayToWorldPointDistanceSq; thisDistSq < prevDistSq)
        intersection = std::move(hit);
    }
  };

  for (const Cesium3DTilesSelection::Tile::Pointer& pTile : additiveCandidateTiles) {
    intersectVisibleTile(pTile.get());
  }

  for (const Cesium3DTilesSelection::Tile::Pointer& pTile : candidateTiles) {
    intersectVisibleTile(pTile.get());
  }

  if (intersection.has_value()) {
    return ellipsoid.getMaximumRadius() * rayOriginHeightFraction -
           glm::sqrt(intersection->rayToWorldPointDistanceSq);
  }

  return 0.0;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool Body::getIntersection(
    glm::dvec3 const& rayPos, glm::dvec3 const& rayDir, glm::dvec3& pos) const {

  auto parent = mSolarSystem->getObject(mName);

  if (!parent || !parent->getIsBodyVisible()) {
    return false;
  }

  auto invTransform = glm::inverse(parent->getObserverRelativeTransform());

  // Transform ray into planet coordinate system.
  glm::dvec4 origin(rayPos, 1.0);
  origin = (invTransform * origin) / glm::dvec4(parent->getRadii(), 1.0);

  glm::dvec4 direction(rayDir, 0.0);
  direction = (invTransform * direction) / glm::dvec4(parent->getRadii(), 1.0);
  direction = glm::normalize(direction);

  double b    = glm::dot(origin.xyz(), direction.xyz());
  double c    = glm::dot(origin.xyz(), origin.xyz()) - 1.0;
  double fDet = b * b - c;

  if (fDet < 0.0) {
    return false;
  }

  fDet = std::sqrt(fDet);
  pos  = (origin + direction * (-b - fDet)).xyz();
  pos *= parent->getRadii();

  return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::cesiumbodies
