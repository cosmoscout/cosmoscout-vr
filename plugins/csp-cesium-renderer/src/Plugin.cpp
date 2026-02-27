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

// CosmoScout headers for the Math Bridge
#include "../../../src/cs-core/SolarSystem.hpp"
#include "../../../src/cs-core/TimeControl.hpp"

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
  // 5. Create the Test Tileset
  std::string testUrl = "https://bertt.github.io/cesium_3dtiles_samples/samples/b3dm/tileset.json";

  mTileset = std::make_unique<Cesium3DTilesSelection::Tileset>(externals, testUrl);
  logger().info("Cesium Externals Assembled. Tileset Created!");

  mRenderer = std::make_unique<CesiumTilesetRenderer>(mTileset.get(), mSolarSystem);
}

void Plugin::deInit() {
  logger().info("Unloading plugin...");

  // Destroy the tileset FIRST — it may have in-flight async operations.
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

  // --- DIAGNOSTIC: Track position over time ---
  static int frameCount = 0;
  frameCount++;

  // Log first 5 frames to see if position changes during init
  if (frameCount <= 5 || frameCount == 100 || frameCount == 500) {
    double cr   = glm::length(camPositionECEF);
    double cLat = std::asin(camPositionECEF.z / cr) * 180.0 / 3.14159265;
    double cLon = std::atan2(camPositionECEF.y, camPositionECEF.x) * 180.0 / 3.14159265;
    logger().info("[DIAG] Frame {} | ObsPos: ({:.0f}, {:.0f}, {:.0f}) | Lat {:.2f}, Lon {:.2f}, "
                  "Alt {:.0f} km",
        frameCount, camPositionECEF.x, camPositionECEF.y, camPositionECEF.z, cLat, cLon,
        (cr - 6371000.0) / 1000.0);
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
