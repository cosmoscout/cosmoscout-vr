////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "Plugin.hpp"
#include "CesiumUtils.hpp"
#include "logger.hpp"

// Cesium headers
#include <Cesium3DTilesContent/registerAllTileContentTypes.h>
#include <Cesium3DTilesSelection/TilesetExternals.h>
#include <Cesium3DTilesSelection/ViewState.h>
#include <CesiumAsync/AsyncSystem.h>
#include <CesiumCurl/CurlAssetAccessor.h>
#include <CesiumUtility/CreditSystem.h>
#include <Cesium3DTilesSelection/TilesetLoadFailureDetails.h>


// CosmoScout headers for the Math Bridge
#include "../../../src/cs-core/SolarSystem.hpp"

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

void Plugin::init() {
  logger().info("Starting Cesium Engine Initialization...");

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
  //5 TilesetOptions — Memory & LOD Configuration
  Cesium3DTilesSelection::TilesetOptions options;
  options.maximumCachedBytes           = 256LL * 1024 * 1024;  // 256 MB cache limit
  options.maximumSimultaneousTileLoads = 20;                    // concurrent downloads
  options.maximumScreenSpaceError      = 16.0;                  // LOD threshold (pixels)
  options.forbidHoles                  = false;                 // faster loading
  options.preloadAncestors             = true;                  // smooth zoom-out
  options.preloadSiblings              = true;                  // smooth panning

  // ── Error Callback — Graceful failure instead of crash ──
  options.loadErrorCallback = [](const Cesium3DTilesSelection::TilesetLoadFailureDetails& details) {
    logger().error("[Cesium] Load FAILED — type: {}, HTTP status: {}, message: {}",
        static_cast<int>(details.type), details.statusCode, details.message);
  };

  // ── Cesium Ion Authentication ──
  int64_t     ionAssetID = 2275207;  // Google Photorealistic 3D Tiles
  std::string ionToken   = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJqdGkiOiI1ZDhhZDZmYi1hYmJhLTRhM2ItODgxNy0wYTBkZjRkNzkwNGIiLCJpZCI6MzkyMzEwLCJpYXQiOjE3NzI1NDM3NDV9.ccVmFT4Ly-_LRLverWw_VETQX-W_Ok1S7EGZIiIDZ_o";

  mTileset = std::make_unique<Cesium3DTilesSelection::Tileset>(
      externals, ionAssetID, ionToken, options);

  logger().info("Cesium Ion Tileset Created (Asset {}). Streaming will begin on first update.", ionAssetID);


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
  // Since observer.center='Earth' and observer.frame='IAU_Earth',
  // observer.getPosition() IS directly the camera ECEF position in meters!
  auto&      observer        = mSolarSystem->getObserver();
  glm::dvec3 camPositionECEF = observer.getPosition();

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
  // Use the rotation from getObserverRelativeTransform to extract the observer orientation.
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
  glm::dvec3 camDirectionECEF = glm::normalize(glm::transpose(rot) * glm::dvec3(0.0, 0.0, -1.0));
  glm::dvec3 camUpECEF        = glm::normalize(glm::transpose(rot) * glm::dvec3(0.0, 1.0, 0.0));

  // 7. Package the ECEF camera into a Cesium ViewState
  //    Hardcoded placeholders for now — will query CosmoScout's GraphicsEngine later.
  glm::dvec2 viewportSize(1920.0, 1080.0);
  double     hFov = glm::radians(45.0);                     // horizontal field of view
  double     vFov = glm::radians(45.0 * (1080.0 / 1920.0)); // maintain aspect ratio

  Cesium3DTilesSelection::ViewState viewState(camPositionECEF, // position (ECEF meters)
      camDirectionECEF,                                        // look direction (ECEF)
      camUpECEF,                                               // up direction (ECEF)
      viewportSize,                                            // viewport in pixels
      hFov,                                                    // horizontal FOV (radians)
      vFov                                                     // vertical FOV (radians)
  );

  // 8. Tell the tileset what the camera sees, then kick off tile loading.
  std::vector<Cesium3DTilesSelection::ViewState> frustums = {viewState};
  mTileset->updateViewGroup(mTileset->getDefaultViewGroup(), frustums);
  mTileset->loadTiles();
}

} // namespace csp::cesiumrenderer
