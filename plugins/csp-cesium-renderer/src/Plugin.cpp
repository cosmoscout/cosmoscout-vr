////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "Plugin.hpp"
#include "CesiumUtils.hpp"
#include "logger.hpp"

// Cesium headers
#include <CesiumAsync/AsyncSystem.h>
#include <CesiumCurl/CurlAssetAccessor.h>
#include <CesiumUtility/CreditSystem.h>
#include <Cesium3DTilesContent/registerAllTileContentTypes.h>
#include <Cesium3DTilesSelection/TilesetExternals.h>
#include <Cesium3DTilesSelection/ViewState.h>

// CosmoScout headers for the Math Bridge
#include "../../../src/cs-core/SolarSystem.hpp"
#include "../../../src/cs-core/TimeControl.hpp"


// ------------------------------------------------------------------------------------------------ //
// (DLL EXPORTS)                                                                    //
// ------------------------------------------------------------------------------------------------ //

EXPORT_FN cs::core::PluginBase* create() {
  return new csp::cesiumrenderer::Plugin;
}

EXPORT_FN void destroy(cs::core::PluginBase* pluginBase) {
  delete pluginBase; // NOLINT(cppcoreguidelines-owning-memory)
}

// ------------------------------------------------------------------------------------------------ //
// THE IMPLEMENTATION                                                                               //
// ------------------------------------------------------------------------------------------------ //

namespace csp::cesiumrenderer {

void Plugin::init() {
    logger().info("Starting Cesium Engine Initialization...");

    // 1. The Binary Parser Fix 
    Cesium3DTilesContent::registerAllTileContentTypes();

    // 2. Instantiate our Custom system
    auto taskProcessor   = std::make_shared<CosmoScoutTaskProcessor>();
    mAsyncSystem  = std::make_shared<CesiumAsync::AsyncSystem>(taskProcessor);
    mCreditSystem = std::make_shared<CesiumUtility::CreditSystem>();
    auto prepareRenderer = std::make_shared<StubPrepareRendererResources>();

    // 3. The 404 Fix: Create Network Downloader with Custom User-Agent
    CesiumCurl::CurlAssetAccessorOptions accessorOptions;
    accessorOptions.userAgent = "CosmoScout Cesium Renderer";
    auto assetAccessor = std::make_shared<CesiumCurl::CurlAssetAccessor>(accessorOptions);

    // 4. Assemble the Engine Core
    Cesium3DTilesSelection::TilesetExternals externals{
        assetAccessor,       // pAssetAccessor
        prepareRenderer,     // pPrepareRendererResources
        *mAsyncSystem,       // asyncSystem (dereferenced — struct takes by VALUE)
        mCreditSystem        // pCreditSystem
    };
  // 5. Create the Test Tileset
    std::string testUrl =
        "https://raw.githubusercontent.com/CesiumGS/3d-tiles-samples/"
        "main/1.0/TilesetWithDiscreteLOD/tileset.json";
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

   // Convert CosmoScout's camera (J2000) to Cesium's frame (ECEF/IAU_Earth).

    // 1. Get the current simulation time (SPICE needs this to compute Earth's rotation)
  double simTime = mTimeControl->pSimulationTime.get();

     // 2. Get the observer (camera) — lives in J2000
  auto& observer = mSolarSystem->getObserver();

  // 3. Create our "Earth anchor" — a temporary reference point at Earth's center,
  //    in the IAU_Earth (body-fixed) frame. This IS the ECEF origin.
  cs::scene::CelestialAnchor earthAnchor("Earth", "IAU_Earth");

  // 4. Ask: "Where is the observer, expressed in Earth's body-fixed coordinates?"
  //    This is the SPICE magic — it handles the J2000 → IAU_Earth rotation internally.
  glm::dvec3 camPositionECEF = earthAnchor.getRelativePosition(simTime, observer);

  // 5. Ask: "What rotation aligns Earth's frame with the observer's frame?"
  //    This rotation R transforms vectors FROM Earth-frame TO observer-frame.
  //    We need the INVERSE to go FROM observer-local TO Earth-frame (ECEF).
  glm::dquat earthToObserver = earthAnchor.getRelativeRotation(simTime, observer);
  glm::dquat observerToEarth = glm::inverse(earthToObserver);

  // 6. Transform the observer's local camera axes into ECEF.
  //    In OpenGL convention: forward = -Z, up = +Y
  glm::dvec3 camDirectionECEF = observerToEarth * glm::dvec3(0.0, 0.0, -1.0);
  glm::dvec3 camUpECEF        = observerToEarth * glm::dvec3(0.0, 1.0,  0.0);


  // 7. Package the ECEF camera into a Cesium ViewState
  //    Hardcoded placeholders for now — will query CosmoScout's GraphicsEngine later.
  glm::dvec2 viewportSize(1920.0, 1080.0);
  double     hFov = glm::radians(45.0);  // horizontal field of view
  double     vFov = glm::radians(45.0 * (1080.0 / 1920.0));  // maintain aspect ratio

  Cesium3DTilesSelection::ViewState viewState(
      camPositionECEF,     // position (ECEF meters)
      camDirectionECEF,    // look direction (ECEF)
      camUpECEF,           // up direction (ECEF)
      viewportSize,        // viewport in pixels
      hFov,                // horizontal FOV (radians)
      vFov                 // vertical FOV (radians)
  );

  // 8. Tell the tileset what the camera sees, then kick off tile loading.
  std::vector<Cesium3DTilesSelection::ViewState> frustums = {viewState};
  mTileset->updateViewGroup(mTileset->getDefaultViewGroup(), frustums);
  mTileset->loadTiles();


}

} // namespace csp::cesiumrenderer
