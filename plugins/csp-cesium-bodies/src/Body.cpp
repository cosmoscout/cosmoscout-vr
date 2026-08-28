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
#include "../../../../src/cs-utils/FrameStats.hpp"
#include "../../../src/cs-utils/convert.hpp"

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

#include <CesiumGeometry/IntersectionTests.h>
#include <CesiumGeometry/Ray.h>
#include <CesiumGltfContent/GltfUtilities.h>

#include <stack>

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
  cs::utils::FrameStats::ScopedTimer timer("update", cs::utils::FrameStats::TimerMode::eCPU);

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

  VistaViewport* viewport = GetVistaSystem()->GetDisplayManager()->GetViewports().begin()->second;
  int            sizeX = 1920, sizeY = 1080; // fallback if query fails
  viewport->GetViewportProperties()->GetSize(sizeX, sizeY);

  // CosmoScout's observer scale magnifies the view: Scale=0.2 means the world appears 5× bigger
  // (1/0.2 = 5). Each pixel therefore covers 1/5th the physical area. For Cesium's SSE formula
  // (SSE = geoError × viewportH / (dist × 2 × tan(vFov/2))), this magnification is equivalent to
  // having a proportionally larger viewport. This is exactly what csp-lod-bodies does implicitly -
  // its LODVisitor extracts the camera from the modelview matrix which has the scale baked in. We
  // replicate this by scaling the viewport.
  // IMPORTANT: Only enlarge the viewport when Scale < 1.0 (magnified/close-up view). AtScale >= 1.0
  // (orbit/far view), the physical viewport correctly represents the screen - the camera IS
  // physically far away and Cesium's SSE is naturally correct.
  double     scaleFactor = std::min(observer.getScale(), 1.0);
  glm::dvec2 viewportSize(
      static_cast<double>(sizeX) / scaleFactor, static_cast<double>(sizeY) / scaleFactor);

  double left = -0.5, right = 0.5, bottom = -0.5, top = 0.5;
  auto*  pProjProps = viewport->GetProjection()->GetProjectionProperties();
  pProjProps->GetProjPlaneExtents(left, right, bottom, top);
  double hFov = 2.0 * std::atan((right - left) / 2.0);
  double vFov = 2.0 * std::atan((top - bottom) / 2.0);

  Cesium3DTilesSelection::ViewState viewState(
      camPositionECEF, camDirectionECEF, camUpECEF, viewportSize, hFov, vFov);

  std::vector frustums  = {viewState};
  mLastViewUpdateResult = mTileset->updateViewGroup(mTileset->getDefaultViewGroup(), frustums);
  mTileset->loadTiles();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static bool boundingVolumeContainsCoordinate(
    Cesium3DTilesSelection::BoundingVolume const& boundingVolume, CesiumGeometry::Ray const& ray,
    CesiumGeospatial::Cartographic const& coordinate,
    CesiumGeospatial::Ellipsoid const&    ellipsoid) {

  return std::visit(
      [&]<typename BV>(BV const& bv) {
        using T = std::decay_t<BV>;
        if constexpr (std::is_same_v<T, CesiumGeometry::OrientedBoundingBox>) {
          return CesiumGeometry::IntersectionTests::rayOBBParametric(ray, bv).value_or(-1) >= 0;
        } else if constexpr (std::is_same_v<T, CesiumGeospatial::BoundingRegion>) {
          return bv.getRectangle().contains(coordinate);
        } else if constexpr (std::is_same_v<T, CesiumGeometry::BoundingSphere>) {
          return CesiumGeometry::IntersectionTests::raySphereParametric(ray, bv).value_or(-1) >= 0;
        } else if constexpr (std::is_same_v<T,
                                 CesiumGeospatial::BoundingRegionWithLooseFittingHeights>) {
          return bv.getBoundingRegion().getRectangle().contains(coordinate);
        } else if constexpr (std::is_same_v<T, CesiumGeospatial::S2CellBoundingVolume>) {
          return bv.computeBoundingRegion(ellipsoid).getRectangle().contains(coordinate);
        } else if constexpr (std::is_same_v<T, CesiumGeometry::BoundingCylinderRegion>) {
          return CesiumGeometry::IntersectionTests::rayOBBParametric(
                     ray, bv.toOrientedBoundingBox())
                     .value_or(-1) >= 0;
        }
      },
      boundingVolume);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// The implementation has been taken from Cesium3DTilesSelection::TilesetHeightQuery and modified to
// just query the tileset in the current state synchronously to avoid additional overhead.
double Body::getHeight(glm::dvec2 lngLat) const {
  cs::utils::FrameStats::ScopedTimer timer("getHeight", cs::utils::FrameStats::TimerMode::eCPU);

  // The ray for height queries starts at this fraction of the ellipsoid max radius above the
  // ellipsoid surface. If a tileset surface is more than this distance above the ellipsoid, it may
  // be missed by height queries. 0.007 is chosen to accomodate Olympus Mons, the tallest peak on
  // Mars. 0.007 is seven-tenths of a percent, or about 44,647 meters for WGS84, well above the
  // highest point on Earth.
  constexpr double rayOriginHeightFraction = 0.007;

  auto ellipsoid = mTileset->getEllipsoid();

  CesiumGeospatial::Cartographic position(lngLat.x, lngLat.y, 0.0);
  CesiumGeospatial::Cartographic start(position.longitude, position.latitude,
      ellipsoid.getMaximumRadius() * rayOriginHeightFraction);
  CesiumGeometry::Ray            ray(
      ellipsoid.cartographicToCartesian(start), -ellipsoid.geodeticSurfaceNormal(start));

  std::vector<Cesium3DTilesSelection::Tile::ConstPointer> candidateTiles{};
  std::vector<Cesium3DTilesSelection::Tile::ConstPointer> additiveCandidateTiles{};

  // Iterative depth-first search over the tile tree using an explicit stack, avoiding the
  // recursion/std::function overhead. Children are pushed in reverse order so they are visited in
  // their original (left-to-right) order.
  std::stack<Cesium3DTilesSelection::Tile::ConstPointer> tilesToVisit;
  if (Cesium3DTilesSelection::Tile::ConstPointer root = mTileset->getRootTile()) {
    tilesToVisit.emplace(root);
  }

  if (mLastHeightTile) {
    tilesToVisit.emplace(mLastHeightTile);
  }

  // A tile with no content bounding volume is treated as always passing.
  auto passesContentBoundingVolume = [&ray, &position, &ellipsoid](
                                         Cesium3DTilesSelection::Tile::ConstPointer const& tile) {
    const auto& bv = tile->getContentBoundingVolume();
    return !bv || boundingVolumeContainsCoordinate(*bv, ray, position, ellipsoid);
  };

  while (!tilesToVisit.empty()) {
    Cesium3DTilesSelection::Tile::ConstPointer tile = tilesToVisit.top();
    tilesToVisit.pop();

    if (tile->getState() == Cesium3DTilesSelection::TileLoadState::Failed) {
      continue;
    }

    if (tile->getChildren().empty()) { // Leaf node, it's a candidate
      if (passesContentBoundingVolume(tile)) {
        candidateTiles.emplace_back(tile);
      }
      continue;
    }

    // Non-leaf: additive refinement means the parent itself can also contribute.
    if (tile->getRefine() == Cesium3DTilesSelection::TileRefine::Add &&
        passesContentBoundingVolume(tile)) {
      additiveCandidateTiles.emplace_back(tile);
    }

    // Push children in reverse so the first child is visited first.
    for (auto const& child : tile->getChildren() | std::views::reverse) {
      if (boundingVolumeContainsCoordinate(child.getBoundingVolume(), ray, position, ellipsoid)) {
        tilesToVisit.emplace(&child);
      }
    }
  }

  std::optional<CesiumGltfContent::GltfUtilities::RayGltfHit> intersection;
  for (auto const& tile :
      std::array{std::span{additiveCandidateTiles}, std::span{candidateTiles}} | std::views::join) {
    auto const* renderContent = tile->getContent().getRenderContent();
    if (!renderContent)
      continue;

    auto [hit, _] = CesiumGltfContent::GltfUtilities::intersectRayGltfModel(
        ray, renderContent->getModel(), true, tile->getTransform());

    // Set ray info to this hit if closer, or the first hit
    if (!intersection.has_value()) {
      intersection    = std::move(hit);
      mLastHeightTile = tile;
    } else if (hit) {
      double prevDistSq = intersection->rayToWorldPointDistanceSq;
      if (double thisDistSq = hit->rayToWorldPointDistanceSq; thisDistSq < prevDistSq) {
        intersection    = std::move(hit);
        mLastHeightTile = tile;
      }
    }
  }

  if (intersection.has_value()) {
    return ellipsoid.getMaximumRadius() * rayOriginHeightFraction -
           glm::sqrt(intersection->rayToWorldPointDistanceSq);
  }

  return 0.0;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Helper: does a bounding volume intersect a ray?  It mirrors the logic used in getHeight (which
// tests against a cartographic coordinate) but ignores the geographic‑region cases, they are
// conservatively accepted.
static bool boundingVolumeIntersectsRay(
    Cesium3DTilesSelection::BoundingVolume const& boundingVolume, CesiumGeometry::Ray const& ray,
    CesiumGeospatial::Ellipsoid const& ellipsoid) {

  return std::visit(
      [&]<typename BV>(BV const& bv) {
        using T = std::decay_t<BV>;
        if constexpr (std::is_same_v<T, CesiumGeometry::BoundingSphere>) {
          return CesiumGeometry::IntersectionTests::raySphereParametric(ray, bv).value_or(-1) >= 0;
        }
        CesiumGeometry::OrientedBoundingBox obb{{}, {}};
        if constexpr (std::is_same_v<T, CesiumGeometry::OrientedBoundingBox>) {
          obb = bv;
        } else if constexpr (std::is_same_v<T, CesiumGeometry::BoundingCylinderRegion>) {
          obb = bv.toOrientedBoundingBox();
        } else if constexpr (std::is_same_v<T, CesiumGeospatial::S2CellBoundingVolume>) {
          obb = bv.computeBoundingRegion(ellipsoid).getBoundingBox();
        } else if constexpr (std::is_same_v<T, CesiumGeospatial::BoundingRegion>) {
          obb = bv.getBoundingBox();
        } else if constexpr (std::is_same_v<T,
                                 CesiumGeospatial::BoundingRegionWithLooseFittingHeights>) {
          obb = bv.getBoundingRegion().getBoundingBox();
        }
        return CesiumGeometry::IntersectionTests::rayOBBParametric(ray, obb).value_or(-1) >= 0;
      },
      boundingVolume);
}

//////////////////////////////////////////////////////////////////////////////

bool Body::getIntersection(
    glm::dvec3 const& rayPos, glm::dvec3 const& rayDir, glm::dvec3& pos) const {
  // Find the body and its transform.  If the body is not visible we abort.
  auto parent = mSolarSystem->getObject(mName);
  if (!parent || !parent->getIsBodyVisible()) {
    return false;
  }

  // Transform the incoming world‑space ray into the body‑centric ECEF coordinate system that the
  // tileset uses.
  glm::dmat4 invTransform = glm::inverse(parent->getObserverRelativeTransform());

  glm::dvec4 origin    = invTransform * glm::dvec4(rayPos, 1.0);
  glm::dvec4 direction = invTransform * glm::dvec4(rayDir, 0.0);

  CesiumGeospatial::Ellipsoid ellipsoid = mTileset->getEllipsoid();
  CesiumGeometry::Ray         cesiumRay(origin.zxy(), glm::normalize(direction.zxy()));

  // Depth‑first walk of the tile tree – identical to the algorithm used in getHeight() – but this
  // time we keep every tile whose *content* bounding volume can be intersected by the ray.
  std::vector<Cesium3DTilesSelection::Tile::ConstPointer> candidateTiles{};
  std::vector<Cesium3DTilesSelection::Tile::ConstPointer> additiveCandidateTiles{};

  // Stack‑based DFS (root + optional last‑height cache)
  std::deque<Cesium3DTilesSelection::Tile::ConstPointer> tilesToVisit;
  if (Cesium3DTilesSelection::Tile::ConstPointer root = mTileset->getRootTile()) {
    tilesToVisit.emplace_back(root);
  }

  if (!mLastViewUpdateResult.tilesToRenderThisFrame.empty()) {
    std::ranges::copy(
        mLastViewUpdateResult.tilesToRenderThisFrame, std::back_inserter(tilesToVisit));
  }

  if (mLastIntersectionTile) {
    tilesToVisit.emplace_back(mLastIntersectionTile);
  }

  while (!tilesToVisit.empty()) {
    auto tile = tilesToVisit.back();
    tilesToVisit.pop_back();

    if (tile->getState() == Cesium3DTilesSelection::TileLoadState::Failed)
      continue;

    // Tile‑level culling
    if (!boundingVolumeIntersectsRay(tile->getBoundingVolume(), cesiumRay, ellipsoid)) {
      continue;
    }

    // Content‑BV test: add any tile that has intersecting geometry
    if (auto cv = tile->getContentBoundingVolume()) {
      if (boundingVolumeIntersectsRay(*cv, cesiumRay, ellipsoid) && tile->isRenderable()) {
        candidateTiles.emplace_back(tile);
      }
    } else if (tile->isRenderable()) {
      // Tiles without a content BV are treated as always intersecting (e.g. empty tiles that only
      // contain children).  Adding them is harmless because the later GLTF test will simply fail,
      // but it avoids a special‑case leaf‑only check.
      candidateTiles.emplace_back(tile);
    }

    // Additive refinement: the tile may also contribute its own geometry
    if (tile->getRefine() == Cesium3DTilesSelection::TileRefine::Add) {
      additiveCandidateTiles.emplace_back(tile);
    }

    // Push children that might intersect the ray (tile BV test already passed)
    for (auto const& child : tile->getChildren() | std::views::reverse) {
      // We already know the parent BV intersects, but the child may be far enough away that its own
      // BV does not. Test it again to avoid descending into unrelated branches.
      if (boundingVolumeIntersectsRay(child.getBoundingVolume(), cesiumRay, ellipsoid)) {
        tilesToVisit.emplace_back(&child);
      }
    }
  }

  // Test the ray against the GLTF models of all candidate tiles and keep the closest hit.
  std::optional<CesiumGltfContent::GltfUtilities::RayGltfHit> closestHit;
  for (auto const& tile :
      std::array{std::span{additiveCandidateTiles}, std::span{candidateTiles}} | std::views::join) {

    auto const* renderContent = tile->getContent().getRenderContent();
    if (!renderContent) {
      continue;
    }

    auto [hit, _] = CesiumGltfContent::GltfUtilities::intersectRayGltfModel(
        cesiumRay, renderContent->getModel(), true, tile->getTransform());

    if (!hit) {
      continue;
    }

    if (!closestHit.has_value() ||
        hit->rayToWorldPointDistanceSq < closestHit->rayToWorldPointDistanceSq) {
      closestHit            = hit;
      mLastIntersectionTile = tile; // cache for next queries
    }
  }

  // Return the world‑space intersection point (ECEF) if we found one.
  if (closestHit.has_value()) {
    pos = closestHit->worldPoint.yzx();
    return true;
  }

  return false;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::cesiumbodies
