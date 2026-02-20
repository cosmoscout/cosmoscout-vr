////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSP_CESIUM_RENDERER_CESIUM_UTILS_HPP
#define CSP_CESIUM_RENDERER_CESIUM_UTILS_HPP

// Belt-and-suspenders: Prevent the deadly Windows 'task' macro collision
#ifdef task
#undef task
#endif

#include <CesiumAsync/ITaskProcessor.h>
#include <Cesium3DTilesSelection/IPrepareRendererResources.h>

namespace csp::cesiumrenderer {

    // 1. THE TASK PROCESSOR
    class CosmoScoutTaskProcessor : public CesiumAsync::ITaskProcessor {
    public:
        CosmoScoutTaskProcessor() = default;
        void startTask(std::function<void()> f) override;
    };

    // 2. THE STUB RENDERER
    class StubPrepareRendererResources : public Cesium3DTilesSelection::IPrepareRendererResources {
    public:
        StubPrepareRendererResources() = default;

        // --- 3D Tiles Geometry Handlers ---
        CesiumAsync::Future<Cesium3DTilesSelection::TileLoadResultAndRenderResources> prepareInLoadThread(
            const CesiumAsync::AsyncSystem& asyncSystem,
            Cesium3DTilesSelection::TileLoadResult&& tileLoadResult,
            const glm::dmat4& transform,
            const std::any& rendererOptions) override;

        void* prepareInMainThread(Cesium3DTilesSelection::Tile& tile, void* pLoadThreadResult) override;
        void free(Cesium3DTilesSelection::Tile& tile, void* pLoadThreadResult, void* pMainThreadResult) noexcept override;

        // --- Raster Overlay Handlers ---
        void* prepareRasterInLoadThread(CesiumGltf::ImageAsset& image, const std::any& rendererOptions) override;
        void* prepareRasterInMainThread(CesiumRasterOverlays::RasterOverlayTile& rasterTile, void* pLoadThreadResult) override;
        void freeRaster(const CesiumRasterOverlays::RasterOverlayTile& rasterTile, void* pLoadThreadResult, void* pMainThreadResult) noexcept override;
        void attachRasterInMainThread(const Cesium3DTilesSelection::Tile& tile, int32_t overlayTextureCoordinateID, const CesiumRasterOverlays::RasterOverlayTile& rasterTile, void* pMainThreadRendererResources, const glm::dvec2& translation, const glm::dvec2& scale) override;
        void detachRasterInMainThread(const Cesium3DTilesSelection::Tile& tile, int32_t overlayTextureCoordinateID, const CesiumRasterOverlays::RasterOverlayTile& rasterTile, void* pMainThreadRendererResources) noexcept override;
    };




} // namespace csp::cesiumrenderer

#endif // CSP_CESIUM_RENDERER_CESIUM_UTILS_HPP