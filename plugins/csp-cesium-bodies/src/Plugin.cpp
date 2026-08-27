////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "Plugin.hpp"
#include "Body.hpp"
#include "PrepareRenderResources.hpp"
#include "logger.hpp"

#include <map>

#include "../../../src/cs-core/Settings.hpp"

#include <Cesium3DTilesContent/registerAllTileContentTypes.h>
#include <Cesium3DTilesSelection/TilesetExternals.h>
#include <Cesium3DTilesSelection/TilesetLoadFailureDetails.h>
#include <CesiumAsync/AsyncSystem.h>
#include <CesiumCurl/CurlAssetAccessor.h>
#include <CesiumUtility/CreditSystem.h>

#include "../../../src/cs-core/SolarSystem.hpp"
#include "../../../src/cs-scene/CelestialObserver.hpp"

#include <VistaKernel/DisplayManager/VistaProjection.h>
#include <VistaKernel/DisplayManager/VistaViewport.h>
#include <VistaKernel/VistaSystem.h>

////////////////////////////////////////////////////////////////////////////////////////////////////

EXPORT_FN cs::core::PluginBase* create() {
  return new csp::cesiumbodies::Plugin;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

EXPORT_FN void destroy(cs::core::PluginBase* pluginBase) {
  delete pluginBase; // NOLINT(cppcoreguidelines-owning-memory)
}

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace csp::cesiumbodies {

////////////////////////////////////////////////////////////////////////////////////////////////////

void from_json(nlohmann::json const& j, Plugin::Settings::CesiumBody& o) {
  cs::core::Settings::deserialize(j, "ionAssetId", o.ionAssetId);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void to_json(nlohmann::json& j, Plugin::Settings::CesiumBody const& o) {
  cs::core::Settings::serialize(j, "ionAssetId", o.ionAssetId);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void from_json(nlohmann::json const& j, Plugin::Settings& o) {
  cs::core::Settings::deserialize(j, "bodies", o.mCesiumBodies);
  cs::core::Settings::deserialize(j, "ionToken", o.mIonToken);
  cs::core::Settings::deserialize(j, "cacheSizeMB", o.mCacheSizeMB);
  cs::core::Settings::deserialize(j, "maxConcurrentDownloads", o.mMaxConcurrentDownloads);
  cs::core::Settings::deserialize(j, "maxScreenSpaceError", o.mMaxScreenSpaceError);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void to_json(nlohmann::json& j, Plugin::Settings const& o) {
  cs::core::Settings::serialize(j, "bodies", o.mCesiumBodies);
  cs::core::Settings::serialize(j, "ionToken", o.mIonToken);
  cs::core::Settings::serialize(j, "cacheSizeMB", o.mCacheSizeMB);
  cs::core::Settings::serialize(j, "maxConcurrentDownloads", o.mMaxConcurrentDownloads);
  cs::core::Settings::serialize(j, "maxScreenSpaceError", o.mMaxScreenSpaceError);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

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

  std::string ionToken = mPluginSettings.mIonToken.value_or(
      "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
      "eyJqdGkiOiI1ZDhhZDZmYi1hYmJhLTRhM2ItODgxNy0wYTBkZjRkNzkwNGIiLCJpZCI6MzkyM"
      "zEwLCJpYXQiOjE3NzI1NDM3NDV9.ccVmFT4Ly-_LRLverWw_VETQX-W_Ok1S7EGZIiIDZ_o");

  for (const auto& [name, body] : mPluginSettings.mCesiumBodies) {
    Cesium3DTilesSelection::TilesetOptions options;
    options.maximumCachedBytes = mPluginSettings.mCacheSizeMB.value_or(256) * 1024LL * 1024LL;
    options.maximumSimultaneousTileLoads = mPluginSettings.mMaxConcurrentDownloads.value_or(20);
    options.maximumScreenSpaceError      = mPluginSettings.mMaxScreenSpaceError.value_or(16.0);
    options.forbidHoles                  = true;
    options.preloadAncestors             = true;
    options.preloadSiblings              = true;
    options.renderTilesUnderCamera       = true;
    options.enableFogCulling             = false;
    options.contentOptions.generateMissingNormalsSmooth = false;

    options.loadErrorCallback =
        [](const Cesium3DTilesSelection::TilesetLoadFailureDetails& details) {
          logger().error("Load FAILED — type: {}, HTTP status: {}, message: {}",
              static_cast<int>(details.type), details.statusCode, details.message);
        };

    options.loadErrorCallback =
        [](const Cesium3DTilesSelection::TilesetLoadFailureDetails& details) {
          logger().error("Load FAILED — type: {}, HTTP status: {}, message: {}",
              static_cast<int>(details.type), details.statusCode, details.message);
        };

    auto cesiumBody = std::make_shared<Body>(
        name, externals, body.ionAssetId, ionToken, options, mSolarSystem, mAllSettings);
    mBodies.emplace(name, cesiumBody);

    auto object = mSolarSystem->getObject(name);

    if (object) {
      object->setSurface(cesiumBody);
      object->setIntersectableObject(cesiumBody);
      logger().info("Registered as CelestialSurface for {}.", name);
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::deInit() {
  logger().info("Unloading plugin...");

  for (const auto& name : mBodies | std::views::keys) {
    auto object = mSolarSystem->getObject(name);
    if (object) {
      object->setSurface(nullptr);
      object->setIntersectableObject(nullptr);
    }
  }

  logger().info("Unloading done.");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::update() {
  mAsyncSystem->dispatchMainThreadTasks();

  auto observer = mSolarSystem->getObserver();
  for (auto& body : mBodies | std::views::values) {
    body->update(observer);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::cesiumbodies
