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


// ------------------------------------------------------------------------------------------------ //
// THE DOOR HANDLE (DLL EXPORTS)                                                                    //
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

    logger().info("Cesium Externals Assembled Successfully!");
}

void Plugin::deInit() {
  logger().info("Unloading plugin...");
    
  // Future cleanup code will go here

  logger().info("Unloading done.");
}

void Plugin::update() {
  // Runs 60 times a second. Do not put logger().info() here or it will freeze the console!
}

} // namespace csp::cesiumrenderer
