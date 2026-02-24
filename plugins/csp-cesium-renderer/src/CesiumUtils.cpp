////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include <GL/glew.h>
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

void* StubPrepareRendererResources::prepareInMainThread(
    Cesium3DTilesSelection::Tile& tile, void* pLoadThreadResult) {

    // --- STEP 1: Cast the void* back to our typed struct ---
    auto* pData = static_cast<CesiumRenderData*>(pLoadThreadResult);

    // If the CPU thread didn't produce any data, nothing to upload.
    if (!pData || pData->vertices.empty()) {
        return nullptr;
    }

    // Save the index count BEFORE we clear the vectors later.
    pData->indexCount = static_cast<uint32_t>(pData->indices.size());

    logger().info("[Cesium] GPU Upload: {} vertices, {} indices.",
        pData->vertices.size() / 6, pData->indexCount);

    // --- STEP 2: Generate the VAO (Vertex Array Object) ---
    glGenVertexArrays(1, &pData->vao);
    glBindVertexArray(pData->vao);

    // --- STEP 3: Generate and fill the VBO (Vertex Buffer Object) ---
    glGenBuffers(1, &pData->vbo);
    glBindBuffer(GL_ARRAY_BUFFER, pData->vbo);
    glBufferData(
        GL_ARRAY_BUFFER,
        pData->vertices.size() * sizeof(float),
        pData->vertices.data(),
        GL_STATIC_DRAW
    );

    // --- STEP 4: Generate and fill the EBO (Element Buffer Object) ---
    glGenBuffers(1, &pData->ebo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, pData->ebo);
    glBufferData(
        GL_ELEMENT_ARRAY_BUFFER,
        pData->indices.size() * sizeof(uint32_t),
        pData->indices.data(),
        GL_STATIC_DRAW
    );

    // --- STEP 5: Define the vertex attribute layout ---
    // Location 0 = Position (3 floats, offset 0 bytes)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    // Location 1 = Normal (3 floats, offset 12 bytes)
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    // --- STEP 6: Unbind the VAO (good hygiene) ---
    glBindVertexArray(0);

    // --- STEP 7: Free CPU-side memory (it's now in VRAM) ---
    pData->vertices.clear();
    pData->vertices.shrink_to_fit();
    pData->indices.clear();
    pData->indices.shrink_to_fit();

    // Return the pointer. Cesium stores this as "pMainThreadResult"
    // and will pass it to free() when the tile is unloaded.
    return pData;
}

void StubPrepareRendererResources::free(
    Cesium3DTilesSelection::Tile& tile,
    void* pLoadThreadResult,
    void* pMainThreadResult) noexcept {

    // Case 1: prepareInMainThread was already called.
    // In that case, pMainThreadResult holds our CesiumRenderData*.
    if (pMainThreadResult) {
        auto* pData = static_cast<CesiumRenderData*>(pMainThreadResult);

        // Delete the GPU resources from VRAM
        if (pData->vao != 0) {
            glDeleteVertexArrays(1, &pData->vao);
        }
        if (pData->vbo != 0) {
            glDeleteBuffers(1, &pData->vbo);
        }
        if (pData->ebo != 0) {
            glDeleteBuffers(1, &pData->ebo);
        }

        logger().debug("[Cesium] GPU Free: Deleted VAO={}, VBO={}, EBO={}.",
            pData->vao, pData->vbo, pData->ebo);

        // Delete the struct itself from CPU heap memory
        delete pData;
    }

    // Case 2: prepareInMainThread was NEVER called.
    // The tile was loaded by the CPU thread but unloaded before the
    // main thread had a chance to upload it. The CPU data still exists.
    if (pLoadThreadResult) {
        auto* pData = static_cast<CesiumRenderData*>(pLoadThreadResult);

        // No GPU resources to delete (they were never created).
        // Just delete the CPU-side struct.
        delete pData;
    }
}
void* StubPrepareRendererResources::prepareRasterInLoadThread(CesiumGltf::ImageAsset& image, const std::any& rendererOptions) { return nullptr; }
void* StubPrepareRendererResources::prepareRasterInMainThread(CesiumRasterOverlays::RasterOverlayTile& rasterTile, void* pLoadThreadResult) { return nullptr; }
void StubPrepareRendererResources::freeRaster(const CesiumRasterOverlays::RasterOverlayTile& rasterTile, void* pLoadThreadResult, void* pMainThreadResult) noexcept {}
void StubPrepareRendererResources::attachRasterInMainThread(const Cesium3DTilesSelection::Tile& tile, int32_t overlayTextureCoordinateID, const CesiumRasterOverlays::RasterOverlayTile& rasterTile, void* pMainThreadRendererResources, const glm::dvec2& translation, const glm::dvec2& scale) {}
void StubPrepareRendererResources::detachRasterInMainThread(const Cesium3DTilesSelection::Tile& tile, int32_t overlayTextureCoordinateID, const CesiumRasterOverlays::RasterOverlayTile& rasterTile, void* pMainThreadRendererResources) noexcept {}

} // namespace csp::cesiumrenderer