////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "Plugin.hpp"
#include "PrepareRenderResources.hpp"
#include "logger.hpp"

#include "../../../src/cs-core/Settings.hpp"

#include <glm/glm.hpp>

#include <Cesium3DTilesContent/registerAllTileContentTypes.h>
#include <Cesium3DTilesSelection/TilesetExternals.h>
#include <Cesium3DTilesSelection/TilesetLoadFailureDetails.h>
#include <Cesium3DTilesSelection/ViewState.h>
#include <CesiumAsync/AsyncSystem.h>
#include <CesiumCurl/CurlAssetAccessor.h>
#include <CesiumUtility/CreditSystem.h>

#include "../../../src/cs-core/SolarSystem.hpp"

#include <VistaKernel/DisplayManager/VistaDisplayManager.h>
#include <VistaKernel/DisplayManager/VistaProjection.h>
#include <VistaKernel/DisplayManager/VistaViewport.h>
#include <VistaKernel/VistaSystem.h>

EXPORT_FN cs::core::PluginBase* create() {
  return new csp::cesiumbodies::Plugin;
}

EXPORT_FN void destroy(cs::core::PluginBase* pluginBase) {
  delete pluginBase; // NOLINT(cppcoreguidelines-owning-memory)
}

namespace csp::cesiumbodies {

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

  from_json(mAllSettings->mPlugins.at("csp-cesium-bodies"), mPluginSettings);

  Cesium3DTilesContent::registerAllTileContentTypes();

  auto taskProcessor   = std::make_shared<TaskProcessor>();
  mAsyncSystem         = std::make_shared<CesiumAsync::AsyncSystem>(taskProcessor);
  mCreditSystem        = std::make_shared<CesiumUtility::CreditSystem>();
  auto prepareRenderer = std::make_shared<PrepareRendererResources>();

  CesiumCurl::CurlAssetAccessorOptions accessorOptions;
  accessorOptions.userAgent = "CosmoScout Cesium Bodies";
  auto assetAccessor        = std::make_shared<CesiumCurl::CurlAssetAccessor>(accessorOptions);

  Cesium3DTilesSelection::TilesetExternals externals{.pAssetAccessor = assetAccessor,
      .pPrepareRendererResources                                     = prepareRenderer,
      .asyncSystem                                                   = *mAsyncSystem,
      .pCreditSystem                                                 = mCreditSystem};

  Cesium3DTilesSelection::TilesetOptions options;
  options.maximumCachedBytes = mPluginSettings.mCacheSizeMB.value_or(256) * 1024LL * 1024LL;
  options.maximumSimultaneousTileLoads = mPluginSettings.mMaxConcurrentDownloads.value_or(20);
  options.maximumScreenSpaceError      = mPluginSettings.mMaxScreenSpaceError.value_or(16.0);
  options.forbidHoles                  = true;
  options.preloadAncestors             = true;
  options.preloadSiblings              = true;
  options.contentOptions.generateMissingNormalsSmooth = true;

  options.loadErrorCallback = [](const Cesium3DTilesSelection::TilesetLoadFailureDetails& details) {
    logger().error("Load FAILED — type: {}, HTTP status: {}, message: {}",
        static_cast<int>(details.type), details.statusCode, details.message);
  };

  int64_t     ionAssetID = mPluginSettings.mIonAssetId.value_or(2275207);
  std::string ionToken   = mPluginSettings.mIonToken.value_or(
      "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
        "eyJqdGkiOiI1ZDhhZDZmYi1hYmJhLTRhM2ItODgxNy0wYTBkZjRkNzkwNGIiLCJpZCI6MzkyM"
        "zEwLCJpYXQiOjE3NzI1NDM3NDV9.ccVmFT4Ly-_LRLverWw_VETQX-W_Ok1S7EGZIiIDZ_o");

  mTileset =
      std::make_unique<Cesium3DTilesSelection::Tileset>(externals, ionAssetID, ionToken, options);

  logger().info(
      "Cesium Ion Tileset Created (Asset {}). Streaming will begin on first update.", ionAssetID);

  mRenderer = std::make_shared<TilesetRenderer>(mTileset.get(), mSolarSystem);

  // TODO: Extract celestial body.
  auto earth = mSolarSystem->getObject("Earth");
  if (earth) {
    earth->setSurface(mRenderer);
    earth->setIntersectableObject(mRenderer);
    logger().info("Registered as CelestialSurface for Earth.");
  }
}

void Plugin::deInit() {
  logger().info("Unloading plugin...");

  // TODO: Extract celestial body.
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
  mAsyncSystem->dispatchMainThreadTasks();

  auto&      observer = mSolarSystem->getObserver();
  glm::dvec3 glmPos   = observer.getPosition();
  glm::dvec3 camPositionECEF(glmPos.zxy());

  // Guard: Skip Cesium updates while the observer is still flying to Earth.
  // On startup, CosmoScout animates the observer from the Solar System Barycenter
  // (~1 AU away) to Earth orbit. During this ~8-second transit, the position is
  // meaningless for Cesium's LOD system. We wait until the camera is within
  // 1,000,000 km of Earth's center.
  if (double camDistFromEarthCenter = glm::length(camPositionECEF); camDistFromEarthCenter > 1e9) {
    return;
  }

  // TODO: Extract celestial body.
  auto earth = mSolarSystem->getObject("Earth");
  if (!earth)
    return;
  glm::dmat4 earthToObserver = earth->getObserverRelativeTransform();
  glm::dmat3 rot;
  if (double s = glm::length(glm::dvec3(earthToObserver[0])); s > 0.0) {
    rot[0] = glm::dvec3(earthToObserver[0]) / s;
    rot[1] = glm::dvec3(earthToObserver[1]) / s;
    rot[2] = glm::dvec3(earthToObserver[2]) / s;
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
  double     scaleFactor = std::max(std::min(observer.getScale(), 1.0), 0.001);
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

} // namespace csp::cesiumbodies
