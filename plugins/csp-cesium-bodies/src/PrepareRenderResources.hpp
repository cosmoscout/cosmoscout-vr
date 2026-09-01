////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSP_CESIUM_BODIES_PREPARE_RENDERER_RESOURCES_HPP
#define CSP_CESIUM_BODIES_PREPARE_RENDERER_RESOURCES_HPP

#include <Cesium3DTilesSelection/IPrepareRendererResources.h>
#include <CesiumAsync/ITaskProcessor.h>
#include <glm/glm.hpp>

namespace csp::cesiumbodies {

class TaskProcessor : public CesiumAsync::ITaskProcessor {
 public:
  TaskProcessor() = default;
  void startTask(std::function<void()> f) override;
};

class PrepareRendererResources : public Cesium3DTilesSelection::IPrepareRendererResources {
 public:
  PrepareRendererResources() = default;

  CesiumAsync::Future<Cesium3DTilesSelection::TileLoadResultAndRenderResources> prepareInLoadThread(
      const CesiumAsync::AsyncSystem&          asyncSystem,
      Cesium3DTilesSelection::TileLoadResult&& tileLoadResult, const glm::dmat4& transform,
      const std::any& rendererOptions) override;

  void* prepareInMainThread(Cesium3DTilesSelection::Tile& tile, void* pLoadThreadResult) override;
  void  free(Cesium3DTilesSelection::Tile& tile, void* loadThreadResult,
       void* pMainThreadResult) noexcept override;

  void* prepareRasterInLoadThread(
      CesiumImage::ImageAsset& image, const std::any& rendererOptions) override;

  void* prepareRasterInMainThread(
      CesiumRasterOverlays::RasterOverlayTile& rasterTile, void* pLoadThreadResult) override;

  void freeRaster(const CesiumRasterOverlays::RasterOverlayTile& rasterTile,
      void* pLoadThreadResult, void* pMainThreadResult) noexcept override;

  void attachRasterInMainThread(const Cesium3DTilesSelection::Tile& tile,
      int32_t overlayTextureCoordinateID, const CesiumRasterOverlays::RasterOverlayTile& rasterTile,
      void* pMainThreadRendererResources, const glm::dvec2& translation,
      const glm::dvec2& scale) override;

  void detachRasterInMainThread(const Cesium3DTilesSelection::Tile& tile,
      int32_t overlayTextureCoordinateID, const CesiumRasterOverlays::RasterOverlayTile& rasterTile,
      void* pMainThreadRendererResources) noexcept override;
};

} // namespace csp::cesiumbodies

#endif // CSP_CESIUM_BODIES_PREPARE_RENDERER_RESOURCES_HPP
