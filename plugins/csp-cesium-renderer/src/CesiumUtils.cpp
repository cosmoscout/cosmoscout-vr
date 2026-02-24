////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "CesiumUtils.hpp"
#include "logger.hpp"
#include <CesiumAsync/AsyncSystem.h>
#include <CesiumGltf/AccessorView.h>
#include <thread>

namespace csp::cesiumrenderer {

void CosmoScoutTaskProcessor::startTask(std::function<void()> f) {
    std::thread(std::move(f)).detach();
}

CesiumAsync::Future<Cesium3DTilesSelection::TileLoadResultAndRenderResources> StubPrepareRendererResources::prepareInLoadThread(
    const CesiumAsync::AsyncSystem& asyncSystem,
    Cesium3DTilesSelection::TileLoadResult&& tileLoadResult,
    [[maybe_unused]] const glm::dmat4& transform,
    const std::any& rendererOptions) {
    
    // --- STEP A: Check if the tile contains a glTF model ---
    CesiumGltf::Model* pModel =
        std::get_if<CesiumGltf::Model>(&tileLoadResult.contentKind);

    if (!pModel) {
        // Not a model (empty tile, external tileset, or unknown)
        logger().debug("[Cesium] CPU Thread: Tile has no model data, skipping.");
        return asyncSystem.createResolvedFuture(
            Cesium3DTilesSelection::TileLoadResultAndRenderResources{
                std::move(tileLoadResult),
                nullptr
            });
    }

    logger().info("[Cesium] CPU Thread: Model found! Meshes: {}", pModel->meshes.size());

    // --- STEP B: Create our render data container on the heap ---
    auto* renderData = new CesiumRenderData();

    // --- STEP C: Loop over every mesh and every primitive ---
    for (const CesiumGltf::Mesh& mesh : pModel->meshes) {
        for (const CesiumGltf::MeshPrimitive& primitive : mesh.primitives) {

            // --- C1: Find the POSITION accessor index ---
            auto posIt = primitive.attributes.find("POSITION");
            if (posIt == primitive.attributes.end()) {
                logger().warn("[Cesium] CPU Thread: Primitive has no POSITION attribute, skipping.");
                continue;
            }
            int32_t positionAccessorIndex = posIt->second;

            // --- C2: Create a typed view into the position data ---
            CesiumGltf::AccessorView<CesiumGltf::AccessorTypes::VEC3<float>> positions(
                *pModel, positionAccessorIndex);

            if (positions.status() != CesiumGltf::AccessorViewStatus::Valid) {
                logger().warn("[Cesium] CPU Thread: POSITION accessor invalid (status {}), skipping.",
                    static_cast<int>(positions.status()));
                continue;
            }

            // --- C3: Find the NORMAL accessor index (optional) ---
            bool hasNormals = false;
            CesiumGltf::AccessorView<CesiumGltf::AccessorTypes::VEC3<float>> normals;
            auto normIt = primitive.attributes.find("NORMAL");
            if (normIt != primitive.attributes.end()) {
                normals = CesiumGltf::AccessorView<CesiumGltf::AccessorTypes::VEC3<float>>(
                    *pModel, normIt->second);
                if (normals.status() == CesiumGltf::AccessorViewStatus::Valid) {
                    hasNormals = true;
                }
            }

            // --- C4: Copy positions and normals into our interleaved buffer ---
            // Layout per vertex: [Px, Py, Pz, Nx, Ny, Nz]  (6 floats)
            size_t vertexStart = renderData->vertices.size() / 6;
            renderData->vertices.reserve(
                renderData->vertices.size() + static_cast<size_t>(positions.size()) * 6);

            for (int64_t i = 0; i < positions.size(); ++i) {
                // Position
                renderData->vertices.push_back(positions[i].value[0]);
                renderData->vertices.push_back(positions[i].value[1]);
                renderData->vertices.push_back(positions[i].value[2]);
                // Normal (or default up-vector if missing)
                if (hasNormals) {
                    renderData->vertices.push_back(normals[i].value[0]);
                    renderData->vertices.push_back(normals[i].value[1]);
                    renderData->vertices.push_back(normals[i].value[2]);
                } else {
                    renderData->vertices.push_back(0.0f);
                    renderData->vertices.push_back(0.0f);
                    renderData->vertices.push_back(1.0f);
                }
            }

            // --- C5: Extract indices ---
            if (primitive.indices >= 0) {
                const CesiumGltf::Accessor* pIndexAccessor =
                    CesiumGltf::Model::getSafe(&pModel->accessors, primitive.indices);

                if (pIndexAccessor) {
                    if (pIndexAccessor->componentType == CesiumGltf::Accessor::ComponentType::UNSIGNED_SHORT) {
                        CesiumGltf::AccessorView<CesiumGltf::AccessorTypes::SCALAR<uint16_t>> indexView(
                            *pModel, primitive.indices);
                        if (indexView.status() == CesiumGltf::AccessorViewStatus::Valid) {
                            for (int64_t i = 0; i < indexView.size(); ++i) {
                                renderData->indices.push_back(
                                    static_cast<uint32_t>(indexView[i].value[0]) + static_cast<uint32_t>(vertexStart));
                            }
                        }
                    } else if (pIndexAccessor->componentType == CesiumGltf::Accessor::ComponentType::UNSIGNED_INT) {
                        CesiumGltf::AccessorView<CesiumGltf::AccessorTypes::SCALAR<uint32_t>> indexView(
                            *pModel, primitive.indices);
                        if (indexView.status() == CesiumGltf::AccessorViewStatus::Valid) {
                            for (int64_t i = 0; i < indexView.size(); ++i) {
                                renderData->indices.push_back(
                                    indexView[i].value[0] + static_cast<uint32_t>(vertexStart));
                            }
                        }
                    } else if (pIndexAccessor->componentType == CesiumGltf::Accessor::ComponentType::UNSIGNED_BYTE) {
                        CesiumGltf::AccessorView<CesiumGltf::AccessorTypes::SCALAR<uint8_t>> indexView(
                            *pModel, primitive.indices);
                        if (indexView.status() == CesiumGltf::AccessorViewStatus::Valid) {
                            for (int64_t i = 0; i < indexView.size(); ++i) {
                                renderData->indices.push_back(
                                    static_cast<uint32_t>(indexView[i].value[0]) + static_cast<uint32_t>(vertexStart));
                            }
                        }
                    }
                }
            }
        } // end for each primitive
    } // end for each mesh

    logger().info("[Cesium] CPU Thread: Extracted {} vertices, {} indices.",
        renderData->vertices.size() / 6, renderData->indices.size());

    return asyncSystem.createResolvedFuture(
        Cesium3DTilesSelection::TileLoadResultAndRenderResources{
            std::move(tileLoadResult),
            renderData  // Passed as void* — Cesium stores this for us
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