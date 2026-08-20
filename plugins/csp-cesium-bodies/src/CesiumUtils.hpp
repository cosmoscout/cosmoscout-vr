////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSP_CESIUM_BODIES_CESIUM_UTILS_HPP
#define CSP_CESIUM_BODIES_CESIUM_UTILS_HPP

#include <Cesium3DTilesSelection/IPrepareRendererResources.h>
#include <CesiumAsync/ITaskProcessor.h>
#include <GL/glew.h>
#include <cstddef>
#include <glm/glm.hpp>
#include <vector>

namespace csp::cesiumbodies {

/// This struct carries extracted mesh data from the CPU worker thread to the main (GPU) thread.
/// It lives on the heap and is passed as void*.
struct CesiumRenderData {
  std::vector<float>    vertices; ///< Interleaved: [Px,Py,Pz, Nx,Ny,Nz, U,V] = 8 floats
  std::vector<uint32_t> indices;  ///< Triangle indices (always uint32_t)

  std::vector<std::byte> texturePixels;
  int32_t                texWidth    = 0;
  int32_t                texHeight   = 0;
  int32_t                texChannels = 4;
  bool                   hasTexture  = false;

  GLuint vao       = 0;
  GLuint vbo       = 0;
  GLuint ebo       = 0;
  GLuint textureId = 0;

  uint32_t indexCount = 0;

  /// Corrected tile-to-ECEF transform (with RTC center + up-axis applied).
  /// The renderer uses this instead of raw pTile->getTransform().
  glm::dmat4 tileTransform{1.0};

  /// CPU-side copies retained for getHeight() / getIntersection() queries.
  /// Only positions + indices are kept — normals, UVs, colors are discarded.
  std::vector<glm::dvec3> cpuPositions;
  std::vector<uint32_t>   cpuIndices;
};

class CosmoScoutTaskProcessor : public CesiumAsync::ITaskProcessor {
 public:
  CosmoScoutTaskProcessor() = default;
  void startTask(std::function<void()> f) override;
};

class StubPrepareRendererResources : public Cesium3DTilesSelection::IPrepareRendererResources {
 public:
  StubPrepareRendererResources() = default;

  CesiumAsync::Future<Cesium3DTilesSelection::TileLoadResultAndRenderResources> prepareInLoadThread(
      const CesiumAsync::AsyncSystem&          asyncSystem,
      Cesium3DTilesSelection::TileLoadResult&& tileLoadResult, const glm::dmat4& transform,
      const std::any& rendererOptions) override;

  void* prepareInMainThread(Cesium3DTilesSelection::Tile& tile, void* pLoadThreadResult) override;
  void  free(Cesium3DTilesSelection::Tile& tile, void* pLoadThreadResult,
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

#endif // CSP_CESIUM_BODIES_CESIUM_UTILS_HPP
