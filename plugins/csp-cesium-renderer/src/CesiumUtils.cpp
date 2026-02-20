////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "CesiumUtils.hpp"
#include "logger.hpp"
#include <CesiumAsync/AsyncSystem.h>
#include <thread>

namespace csp::cesiumrenderer {

void CosmoScoutTaskProcessor::startTask(std::function<void()> f) {
    std::thread(std::move(f)).detach();
}

CesiumAsync::Future<Cesium3DTilesSelection::TileLoadResultAndRenderResources> StubPrepareRendererResources::prepareInLoadThread(
    const CesiumAsync::AsyncSystem& asyncSystem,
    Cesium3DTilesSelection::TileLoadResult&& tileLoadResult,
    const glm::dmat4& transform,
    const std::any& rendererOptions) {
    
    logger().info("[Cesium] CPU Thread: Received tile data!");
    
    return asyncSystem.createResolvedFuture(
        Cesium3DTilesSelection::TileLoadResultAndRenderResources{
            std::move(tileLoadResult),
            nullptr
        });
}

void* StubPrepareRendererResources::prepareInMainThread(Cesium3DTilesSelection::Tile& tile, void* pLoadThreadResult) {
    logger().info("[Cesium] GPU Thread: Ready to upload tile!");
    return nullptr;
}

void StubPrepareRendererResources::free(Cesium3DTilesSelection::Tile& tile, void* pLoadThreadResult, void* pMainThreadResult) noexcept {}
void* StubPrepareRendererResources::prepareRasterInLoadThread(CesiumGltf::ImageAsset& image, const std::any& rendererOptions) { return nullptr; }
void* StubPrepareRendererResources::prepareRasterInMainThread(CesiumRasterOverlays::RasterOverlayTile& rasterTile, void* pLoadThreadResult) { return nullptr; }
void StubPrepareRendererResources::freeRaster(const CesiumRasterOverlays::RasterOverlayTile& rasterTile, void* pLoadThreadResult, void* pMainThreadResult) noexcept {}
void StubPrepareRendererResources::attachRasterInMainThread(const Cesium3DTilesSelection::Tile& tile, int32_t overlayTextureCoordinateID, const CesiumRasterOverlays::RasterOverlayTile& rasterTile, void* pMainThreadRendererResources, const glm::dvec2& translation, const glm::dvec2& scale) {}
void StubPrepareRendererResources::detachRasterInMainThread(const Cesium3DTilesSelection::Tile& tile, int32_t overlayTextureCoordinateID, const CesiumRasterOverlays::RasterOverlayTile& rasterTile, void* pMainThreadRendererResources) noexcept {}

} // namespace csp::cesiumrenderer