////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "Plugin.hpp"
#include "CesiumUtils.hpp"
#include "logger.hpp"

// CosmoScout Settings — for JSON config deserialization (nlohmann::json)
#include "../../../src/cs-core/Settings.hpp"

// Cesium headers
#include <Cesium3DTilesContent/registerAllTileContentTypes.h>
#include <Cesium3DTilesSelection/TilesetExternals.h>
#include <Cesium3DTilesSelection/TilesetLoadFailureDetails.h>
#include <Cesium3DTilesSelection/ViewState.h>
#include <CesiumAsync/AsyncSystem.h>
#include <CesiumCurl/CurlAssetAccessor.h>
#include <CesiumUtility/CreditSystem.h>

// CosmoScout headers for the Math Bridge
#include "../../../src/cs-core/SolarSystem.hpp"
#include "../../../src/cs-utils/convert.hpp"

// ViSTA headers for real viewport/projection extraction
#include <VistaKernel/DisplayManager/VistaDisplayManager.h>
#include <VistaKernel/DisplayManager/VistaProjection.h>
#include <VistaKernel/DisplayManager/VistaViewport.h>
#include <VistaKernel/VistaSystem.h>

// ------------------------------------------------------------------------------------------------
// // (DLL EXPORTS)                                                                    //
// ------------------------------------------------------------------------------------------------
// //

EXPORT_FN cs::core::PluginBase* create() {
  return new csp::cesiumrenderer::Plugin;
}

EXPORT_FN void destroy(cs::core::PluginBase* pluginBase) {
  delete pluginBase; // NOLINT(cppcoreguidelines-owning-memory)
}

// ------------------------------------------------------------------------------------------------
// // THE IMPLEMENTATION //
// ------------------------------------------------------------------------------------------------
// //

namespace csp::cesiumrenderer {

// ── JSON ↔ Settings conversion (follows the csp-rings / csp-atmospheres pattern) ──

void from_json(nlohmann::json const& j, Plugin::Settings& o) {
  cs::core::Settings::deserialize(j, "ionAssetId", o.mIonAssetId);
  cs::core::Settings::deserialize(j, "ionToken", o.mIonToken);
  cs::core::Settings::deserialize(j, "cacheSizeMB", o.mCacheSizeMB);
  cs::core::Settings::deserialize(j, "maxConcurrentDownloads", o.mMaxConcurrentDownloads);
  cs::core::Settings::deserialize(j, "maxScreenSpaceError", o.mMaxScreenSpaceError);
}

void to_json(nlohmann::json& j, Plugin::Settings const& o) {
  cs::core::Settings::serialize(j, "ionAssetId", o.mIonAssetId);
  cs::core::Settings::serialize(j, "ionToken", o.mIonToken);
  cs::core::Settings::serialize(j, "cacheSizeMB", o.mCacheSizeMB);
  cs::core::Settings::serialize(j, "maxConcurrentDownloads", o.mMaxConcurrentDownloads);
  cs::core::Settings::serialize(j, "maxScreenSpaceError", o.mMaxScreenSpaceError);
}

void Plugin::init() {
  logger().info("Starting Cesium Engine Initialization...");

  // 0. Read plugin configuration from the scene JSON.
  //    All fields are std::optional — missing keys gracefully fall back to defaults.
  from_json(mAllSettings->mPlugins.at("csp-cesium-renderer"), mPluginSettings);

  // 1. The Binary Parser Fix
  Cesium3DTilesContent::registerAllTileContentTypes();

  // 2. Instantiate our Custom system
  auto taskProcessor   = std::make_shared<CosmoScoutTaskProcessor>();
  mAsyncSystem         = std::make_shared<CesiumAsync::AsyncSystem>(taskProcessor);
  mCreditSystem        = std::make_shared<CesiumUtility::CreditSystem>();
  auto prepareRenderer = std::make_shared<StubPrepareRendererResources>();

  // 3. The 404 Fix: Create Network Downloader with Custom User-Agent
  CesiumCurl::CurlAssetAccessorOptions accessorOptions;
  accessorOptions.userAgent = "CosmoScout Cesium Renderer";
  auto assetAccessor        = std::make_shared<CesiumCurl::CurlAssetAccessor>(accessorOptions);

  // 4. Assemble the Engine Core
  Cesium3DTilesSelection::TilesetExternals externals{
      assetAccessor,   // pAssetAccessor
      prepareRenderer, // pPrepareRendererResources
      *mAsyncSystem,   // asyncSystem (dereferenced — struct takes by VALUE)
      mCreditSystem    // pCreditSystem
  };
  // 5 TilesetOptions — Memory & LOD Configuration
  //   Values come from JSON config (value_or provides the original hardcoded defaults).
  Cesium3DTilesSelection::TilesetOptions options;
  options.maximumCachedBytes = mPluginSettings.mCacheSizeMB.value_or(256) * 1024LL * 1024LL;
  options.maximumSimultaneousTileLoads = mPluginSettings.mMaxConcurrentDownloads.value_or(20);
  options.maximumScreenSpaceError      = mPluginSettings.mMaxScreenSpaceError.value_or(16.0);
  options.forbidHoles                  = false;               // faster loading
  options.preloadAncestors             = true;                // smooth zoom-out
  options.preloadSiblings              = true;                // smooth panning
  options.contentOptions.generateMissingNormalsSmooth = true; // generate normals if absent

  // ── Error Callback — Graceful failure instead of crash ──
  options.loadErrorCallback = [](const Cesium3DTilesSelection::TilesetLoadFailureDetails& details) {
    logger().error("[Cesium] Load FAILED — type: {}, HTTP status: {}, message: {}",
        static_cast<int>(details.type), details.statusCode, details.message);
  };

  // ── Cesium Ion Authentication (read from JSON, fall back to defaults) ──
  int64_t     ionAssetID = mPluginSettings.mIonAssetId.value_or(2275207);
  std::string ionToken   = mPluginSettings.mIonToken.value_or(
        "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
          "eyJqdGkiOiI1ZDhhZDZmYi1hYmJhLTRhM2ItODgxNy0wYTBkZjRkNzkwNGIiLCJpZCI6MzkyM"
          "zEwLCJpYXQiOjE3NzI1NDM3NDV9.ccVmFT4Ly-_LRLverWw_VETQX-W_Ok1S7EGZIiIDZ_o");

  mTileset =
      std::make_unique<Cesium3DTilesSelection::Tileset>(externals, ionAssetID, ionToken, options);

  logger().info(
      "Cesium Ion Tileset Created (Asset {}). Streaming will begin on first update.", ionAssetID);

  mRenderer = std::make_shared<CesiumTilesetRenderer>(mTileset.get(), mSolarSystem);

  // Register our renderer as the Earth's terrain surface and intersectable object.
  // This tells CosmoScout's collision, measurement, and ground-following systems
  // to query OUR geometry for height data and ray intersections.
  auto earth = mSolarSystem->getObject("Earth");
  if (earth) {
    earth->setSurface(mRenderer);
    earth->setIntersectableObject(mRenderer);
    logger().info("Registered as CelestialSurface for Earth.");
  }
}

void Plugin::deInit() {
  logger().info("Unloading plugin...");

  // Unregister from CosmoScout FIRST — remove our surface and intersectable
  // references so the engine doesn't call getHeight() on a destroyed object.
  auto earth = mSolarSystem->getObject("Earth");
  if (earth) {
    earth->setSurface(nullptr);
    earth->setIntersectableObject(nullptr);
  }

  // Destroy the tileset — it may have in-flight async operations.
  // reset() calls the Tileset destructor, which cancels pending downloads
  // and unloads all tile content from memory.
  mRenderer.reset();
  mTileset.reset();

  logger().info("Unloading done.");
}

void Plugin::update() {
  // Pump Cesium's async task queue. Any completed downloads (tileset.json,
  // tile content, etc.) have their continuation callbacks executed HERE,
  // on the main thread, making them safe to touch OpenGL state.
  mAsyncSystem->dispatchMainThreadTasks();

  // --- Convert CosmoScout's camera to Cesium's ECEF frame ---
  // observer.getPosition() returns the camera position in CosmoScout's swizzled GLM frame
  // (GLM-X=90°E, GLM-Y=North, GLM-Z=PrimeMeridian — see CelestialAnchor.cpp:99-110).
  // Cesium expects standard ECEF (X=PrimeMeridian, Y=90°E, Z=North).
  // The inverse permutation: ECEF(X,Y,Z) = GLM(Z,X,Y)
  auto&      observer = mSolarSystem->getObserver();
  glm::dvec3 glmPos   = observer.getPosition();
  glm::dvec3 camPositionECEF(glmPos.z, glmPos.x, glmPos.y);

  // Guard: Skip Cesium updates while the observer is still flying to Earth.
  // On startup, CosmoScout animates the observer from the Solar System Barycenter
  // (~1 AU away) to Earth orbit. During this ~8-second transit, the position is
  // meaningless for Cesium's LOD system. We wait until the camera is within
  // 100,000 km of Earth's center (well beyond geostationary orbit at 42,164 km).
  double camDistFromEarthCenter = glm::length(camPositionECEF);
  if (camDistFromEarthCenter > 1e8) { // > 100,000 km — still in transit
    return;
  }

  // Camera direction and up in ECEF.
  // Extract orientation from the observer-relative transform, then convert the
  // resulting GLM-frame vectors to ECEF with the same (Z,X,Y) permutation.
  auto earth = mSolarSystem->getObject("Earth");
  if (!earth)
    return;
  glm::dmat4 earthToObserver = earth->getObserverRelativeTransform();
  glm::dmat3 rot;
  double     s = glm::length(glm::dvec3(earthToObserver[0]));
  if (s > 0.0) {
    rot[0] = glm::dvec3(earthToObserver[0]) / s;
    rot[1] = glm::dvec3(earthToObserver[1]) / s;
    rot[2] = glm::dvec3(earthToObserver[2]) / s;
  } else {
    rot = glm::dmat3(1.0);
  }
  // Compute direction/up in CosmoScout's GLM frame first
  glm::dvec3 glmDir = glm::normalize(glm::transpose(rot) * glm::dvec3(0.0, 0.0, -1.0));
  glm::dvec3 glmUp  = glm::normalize(glm::transpose(rot) * glm::dvec3(0.0, 1.0, 0.0));
  // Apply inverse permutation: ECEF(X,Y,Z) = GLM(Z,X,Y)
  glm::dvec3 camDirectionECEF(glmDir.z, glmDir.x, glmDir.y);
  glm::dvec3 camUpECEF(glmUp.z, glmUp.x, glmUp.y);

  // 7. Package the ECEF camera into a Cesium ViewState
  //    Extract REAL viewport size and FOV from ViSTA's display manager.
  //    This is critical: Cesium uses these for frustum culling, SSE, and LOD refinement.
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
  double     scaleFactor = std::max(std::min(observer.getScale(), 1.0), 0.001);
  glm::dvec2 viewportSize(
      static_cast<double>(sizeX) / scaleFactor, static_cast<double>(sizeY) / scaleFactor);

  // Extract real FOV from ViSTA's projection plane extents.
  // ViSTA uses SetProjPlaneExtents(left, right, bottom, top) with midpoint at z=-1,
  // so FOV = 2 * atan(halfExtent / projDistance). projDistance = 1.0 by default.
  double left = -0.5, right = 0.5, bottom = -0.5, top = 0.5;
  auto*  pProjProps = pViewport->GetProjection()->GetProjectionProperties();
  pProjProps->GetProjPlaneExtents(left, right, bottom, top);
  double hFov = 2.0 * std::atan((right - left) / 2.0);
  double vFov = 2.0 * std::atan((top - bottom) / 2.0);

  // --- Diagnostic V3: Replicate EXACT updateSceneScale() math ---
  // Uses cartesianToLngLatHeight for true geodetic altitude (not rough maxRadius approximation).
  static int sFrameCount = 0;
  if (++sFrameCount % 300 == 1) {
    auto   diagRadii = earth->getRadii();
    auto   diagLLH   = cs::utils::convert::cartesianToLngLatHeight(camPositionECEF, diagRadii);
    double geodeticH = diagLLH.z;                           // True geodetic height above ellipsoid
    bool   collision = geodeticH < 0.5 && true;             // Earth mIsCollidable=true by default
    double scrollDisplacement = 0.48 * observer.getScale(); // per-frame scroll movement
    logger().warn("[CESIUM_DIAG_V3] Scale={:.4e}, GeodeticH={:.2f}m, Collision={}, "
                  "ScrollDisp={:.2f}m, ScaleFactor={:.4e}, Viewport={:.0f}x{:.0f}",
        observer.getScale(), geodeticH, collision ? "YES" : "no", scrollDisplacement, scaleFactor,
        viewportSize.x, viewportSize.y);
  }

  // Pass the RAW observer ECEF position to Cesium — no position hacks.
  // The viewport scaling above already handles LOD magnification.
  Cesium3DTilesSelection::ViewState viewState(camPositionECEF, // position (raw ECEF)
      camDirectionECEF,                                        // look direction (ECEF)
      camUpECEF,                                               // up direction (ECEF)
      viewportSize,                                            // viewport (scale-adjusted pixels)
      hFov,                                                    // horizontal FOV (radians)
      vFov                                                     // vertical FOV (radians)
  );

  // 8. Tell the tileset what the camera sees, then kick off tile loading.
  std::vector<Cesium3DTilesSelection::ViewState> frustums = {viewState};
  mTileset->updateViewGroup(mTileset->getDefaultViewGroup(), frustums);
  mTileset->loadTiles();
}

} // namespace csp::cesiumrenderer
