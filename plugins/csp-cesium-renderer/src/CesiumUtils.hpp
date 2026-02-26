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

#include <GL/glew.h>
#include <CesiumAsync/ITaskProcessor.h>
#include <Cesium3DTilesSelection/IPrepareRendererResources.h>
#include <vector>
#include <cstdint>
#include <cstddef>  // for std::byte


namespace csp::cesiumrenderer {

        // 0. CPU-SIDE RENDER DATA CONTAINER
    // This struct carries extracted mesh data from the CPU worker thread
    // to the main (GPU) thread. It lives on the heap and is passed as void*.
    struct CesiumRenderData {
    std::vector<float>    vertices; // Interleaved: [Px,Py,Pz, Nx,Ny,Nz, U,V] = 8 floats
    std::vector<uint32_t> indices;  // Triangle indices (always uint32_t)

    // --- TEXTURE CPU DATA (filled by prepareInLoadThread) ---
    std::vector<std::byte> texturePixels; // Raw decoded RGBA/RGB bytes from ImageAsset
    int32_t texWidth    = 0;
    int32_t texHeight   = 0;
    int32_t texChannels = 4;  // 1=R, 2=RG, 3=RGB, 4=RGBA
    bool    hasTexture  = false;

    // GPU handles (filled by prepareInMainThread, cleaned by free)
    GLuint vao = 0;  // Vertex Array Object — the "recipe card"
    GLuint vbo = 0;  // Vertex Buffer Object — vertex data in VRAM
    GLuint ebo = 0;  // Element Buffer Object — index data in VRAM
    GLuint textureId = 0; // GPU texture handle

    // How many indices to draw (saved before we clear the CPU vector)
    uint32_t indexCount = 0;
};


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