////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "CesiumUtils.hpp"
#include "logger.hpp"
#include <CesiumAsync/AsyncSystem.h>
#include <CesiumGltf/AccessorView.h>
#include <CesiumGltf/Image.h>
#include <CesiumGltf/Material.h>
#include <CesiumGltf/Texture.h>
#include <GL/glew.h>

#include <thread>

namespace csp::cesiumrenderer {

void CosmoScoutTaskProcessor::startTask(std::function<void()> f) {
  std::thread(std::move(f)).detach();
}

CesiumAsync::Future<Cesium3DTilesSelection::TileLoadResultAndRenderResources>
StubPrepareRendererResources::prepareInLoadThread(const CesiumAsync::AsyncSystem& asyncSystem,
    Cesium3DTilesSelection::TileLoadResult&&                                      tileLoadResult,
    [[maybe_unused]] const glm::dmat4& transform, const std::any& rendererOptions) {

  // --- STEP A: Check if the tile contains a glTF model ---
  CesiumGltf::Model* pModel = std::get_if<CesiumGltf::Model>(&tileLoadResult.contentKind);

  if (!pModel) {
    // Not a model (empty tile, external tileset, or unknown)
    logger().debug("[Cesium] CPU Thread: Tile has no model data, skipping.");
    return asyncSystem.createResolvedFuture(
        Cesium3DTilesSelection::TileLoadResultAndRenderResources{
            std::move(tileLoadResult), nullptr});
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
      bool                                                             hasNormals = false;
      CesiumGltf::AccessorView<CesiumGltf::AccessorTypes::VEC3<float>> normals;
      auto normIt = primitive.attributes.find("NORMAL");
      if (normIt != primitive.attributes.end()) {
        normals = CesiumGltf::AccessorView<CesiumGltf::AccessorTypes::VEC3<float>>(
            *pModel, normIt->second);
        if (normals.status() == CesiumGltf::AccessorViewStatus::Valid) {
          hasNormals = true;
        }
      }

      // --- C3b: Find the TEXCOORD_0 accessor index (optional) ---
      bool                                                             hasUVs = false;
      CesiumGltf::AccessorView<CesiumGltf::AccessorTypes::VEC2<float>> uvs;
      auto uvIt = primitive.attributes.find("TEXCOORD_0");
      if (uvIt != primitive.attributes.end()) {
        uvs =
            CesiumGltf::AccessorView<CesiumGltf::AccessorTypes::VEC2<float>>(*pModel, uvIt->second);
        if (uvs.status() == CesiumGltf::AccessorViewStatus::Valid) {
          hasUVs = true;
        }
      }

      // --- C4: Copy positions, normals, UVs, and color into our interleaved buffer ---
      // Layout per vertex: [Px, Py, Pz, Nx, Ny, Nz, U, V, R, G, B, A]  (12 floats)

      // --- C4b: Get this primitive's baseColorFactor (once per primitive) ---
      float matR = 0.8f, matG = 0.8f, matB = 0.8f, matA = 1.0f; // grey default
      if (primitive.material >= 0) {
        const auto* pMat = CesiumGltf::Model::getSafe(&pModel->materials, primitive.material);
        if (pMat && pMat->pbrMetallicRoughness) {
          const auto& factor = pMat->pbrMetallicRoughness->baseColorFactor;
          if (factor.size() >= 4) {
            matR = static_cast<float>(factor[0]);
            matG = static_cast<float>(factor[1]);
            matB = static_cast<float>(factor[2]);
            matA = static_cast<float>(factor[3]);
          }
        }
      }

      size_t vertexStart = renderData->vertices.size() / 12;
      renderData->vertices.reserve(
          renderData->vertices.size() + static_cast<size_t>(positions.size()) * 12);
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
        // UV (or default 0,0 if missing)
        if (hasUVs) {
          renderData->vertices.push_back(uvs[i].value[0]);
          renderData->vertices.push_back(uvs[i].value[1]);
        } else {
          renderData->vertices.push_back(0.0f);
          renderData->vertices.push_back(0.0f);
        }
        // Per-vertex color from material baseColorFactor
        renderData->vertices.push_back(matR);
        renderData->vertices.push_back(matG);
        renderData->vertices.push_back(matB);
        renderData->vertices.push_back(matA);
      }

      // --- C5: Extract indices ---
      if (primitive.indices >= 0) {
        const CesiumGltf::Accessor* pIndexAccessor =
            CesiumGltf::Model::getSafe(&pModel->accessors, primitive.indices);

        if (pIndexAccessor) {
          if (pIndexAccessor->componentType ==
              CesiumGltf::Accessor::ComponentType::UNSIGNED_SHORT) {
            CesiumGltf::AccessorView<CesiumGltf::AccessorTypes::SCALAR<uint16_t>> indexView(
                *pModel, primitive.indices);
            if (indexView.status() == CesiumGltf::AccessorViewStatus::Valid) {
              for (int64_t i = 0; i < indexView.size(); ++i) {
                renderData->indices.push_back(static_cast<uint32_t>(indexView[i].value[0]) +
                                              static_cast<uint32_t>(vertexStart));
              }
            }
          } else if (pIndexAccessor->componentType ==
                     CesiumGltf::Accessor::ComponentType::UNSIGNED_INT) {
            CesiumGltf::AccessorView<CesiumGltf::AccessorTypes::SCALAR<uint32_t>> indexView(
                *pModel, primitive.indices);
            if (indexView.status() == CesiumGltf::AccessorViewStatus::Valid) {
              for (int64_t i = 0; i < indexView.size(); ++i) {
                renderData->indices.push_back(
                    indexView[i].value[0] + static_cast<uint32_t>(vertexStart));
              }
            }
          } else if (pIndexAccessor->componentType ==
                     CesiumGltf::Accessor::ComponentType::UNSIGNED_BYTE) {
            CesiumGltf::AccessorView<CesiumGltf::AccessorTypes::SCALAR<uint8_t>> indexView(
                *pModel, primitive.indices);
            if (indexView.status() == CesiumGltf::AccessorViewStatus::Valid) {
              for (int64_t i = 0; i < indexView.size(); ++i) {
                renderData->indices.push_back(static_cast<uint32_t>(indexView[i].value[0]) +
                                              static_cast<uint32_t>(vertexStart));
              }
            }
          }
        }
      }
      // --- C6: Extract texture pixel data (first texture wins) ---
      if (!renderData->hasTexture && primitive.material >= 0) {
        // Level 1: Get the Material from the model
        const CesiumGltf::Material* pMaterial =
            CesiumGltf::Model::getSafe(&pModel->materials, primitive.material);

        if (pMaterial && pMaterial->pbrMetallicRoughness) {
          // Level 2: Get the baseColorTexture info
          const auto& pbr = *pMaterial->pbrMetallicRoughness;

          if (pbr.baseColorTexture) {
            int32_t textureIndex = pbr.baseColorTexture->index;

            // Level 3: Get the Texture object
            const CesiumGltf::Texture* pTexture =
                CesiumGltf::Model::getSafe(&pModel->textures, textureIndex);

            if (pTexture && pTexture->source >= 0) {
              // Level 4: Get the Image object
              const CesiumGltf::Image* pImage =
                  CesiumGltf::Model::getSafe(&pModel->images, pTexture->source);

              if (pImage && pImage->pAsset) {
                // Level 5: Get the ImageAsset (the actual pixels!)
                const CesiumGltf::ImageAsset& asset = *pImage->pAsset;

                if (!asset.pixelData.empty()) {
                  // Level 6: Copy the pixel bytes into our struct
                  renderData->texturePixels = asset.pixelData;
                  renderData->texWidth      = asset.width;
                  renderData->texHeight     = asset.height;
                  renderData->texChannels   = asset.channels;
                  renderData->hasTexture    = true;

                  logger().info(
                      "[Cesium] CPU Thread: Texture extracted: {}x{}, {} channels, {} bytes.",
                      asset.width, asset.height, asset.channels, asset.pixelData.size());
                }
              }
            }
          }
        }
      }

    } // end for each primitive
  } // end for each mesh

  logger().info("[Cesium] CPU Thread: Extracted {} vertices, {} indices.",
      renderData->vertices.size() / 12, renderData->indices.size());

  return asyncSystem.createResolvedFuture(Cesium3DTilesSelection::TileLoadResultAndRenderResources{
      std::move(tileLoadResult),
      renderData // Passed as void* — Cesium stores this for us
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

  logger().info("[Cesium] GPU Upload: {} vertices, {} indices.", pData->vertices.size() / 12,
      pData->indexCount);

  // --- STEP 2: Generate the VAO (Vertex Array Object) ---
  glGenVertexArrays(1, &pData->vao);
  glBindVertexArray(pData->vao);

  // --- STEP 3: Generate and fill the VBO (Vertex Buffer Object) ---
  glGenBuffers(1, &pData->vbo);
  glBindBuffer(GL_ARRAY_BUFFER, pData->vbo);
  glBufferData(GL_ARRAY_BUFFER, pData->vertices.size() * sizeof(float), pData->vertices.data(),
      GL_STATIC_DRAW);

  // --- STEP 4: Generate and fill the EBO (Element Buffer Object) ---
  glGenBuffers(1, &pData->ebo);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, pData->ebo);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, pData->indices.size() * sizeof(uint32_t),
      pData->indices.data(), GL_STATIC_DRAW);

  // --- STEP 5: Define the vertex attribute layout ---
  // Stride = 12 floats = 48 bytes per vertex: [Px,Py,Pz, Nx,Ny,Nz, U,V, R,G,B,A]
  GLsizei stride = 12 * sizeof(float);

  // Location 0 = Position (3 floats, offset 0 bytes)
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, (void*)0);
  glEnableVertexAttribArray(0);

  // Location 1 = Normal (3 floats, offset 12 bytes)
  glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, (void*)(3 * sizeof(float)));
  glEnableVertexAttribArray(1);

  // Location 2 = UV (2 floats, offset 24 bytes)
  glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, stride, (void*)(6 * sizeof(float)));
  glEnableVertexAttribArray(2);

  // Location 3 = Color (4 floats, offset 32 bytes)
  glVertexAttribPointer(3, 4, GL_FLOAT, GL_FALSE, stride, (void*)(8 * sizeof(float)));
  glEnableVertexAttribArray(3);

  // --- STEP 6: Unbind the VAO (good hygiene) ---
  glBindVertexArray(0);

  // --- STEP 6b: Upload texture to GPU (if we have one) ---
  if (pData->hasTexture && !pData->texturePixels.empty()) {
    // Determine the OpenGL pixel format based on channel count
    GLenum format = GL_RGBA;
    if (pData->texChannels == 1)
      format = GL_RED;
    else if (pData->texChannels == 2)
      format = GL_RG;
    else if (pData->texChannels == 3)
      format = GL_RGB;

    // Create the GPU texture object
    glGenTextures(1, &pData->textureId);
    glBindTexture(GL_TEXTURE_2D, pData->textureId);

    // CRITICAL: Cesium pixels are tightly packed — no row padding
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

    // Upload the raw pixel bytes from CPU RAM to VRAM
    glTexImage2D(GL_TEXTURE_2D,     // target
        0,                          // mip level 0 (the full-size base image)
        format,                     // internal format (how GPU stores it)
        pData->texWidth,            // width in pixels
        pData->texHeight,           // height in pixels
        0,                          // border (always 0, legacy parameter)
        format,                     // pixel data format (how OUR bytes are laid out)
        GL_UNSIGNED_BYTE,           // each channel is one byte (0-255)
        pData->texturePixels.data() // pointer to the raw bytes
    );

    // Restore OpenGL default alignment to avoid contaminating other code
    glPixelStorei(GL_UNPACK_ALIGNMENT, 4);

    // Generate mipmaps for better quality at distance
    glGenerateMipmap(GL_TEXTURE_2D);

    // Set texture filtering and wrapping parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    // Unbind texture (good hygiene)
    glBindTexture(GL_TEXTURE_2D, 0);

    logger().info("[Cesium] GPU Upload: Texture uploaded, ID={}, {}x{}.", pData->textureId,
        pData->texWidth, pData->texHeight);

    // Free CPU-side pixel data immediately (it's now in VRAM)
    pData->texturePixels.clear();
    pData->texturePixels.shrink_to_fit();
  }

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
    Cesium3DTilesSelection::Tile& tile, void* pLoadThreadResult, void* pMainThreadResult) noexcept {

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
    if (pData->textureId != 0) {
      glDeleteTextures(1, &pData->textureId);
    }

    logger().debug("[Cesium] GPU Free: Deleted VAO={}, VBO={}, EBO={}, Tex={}.", pData->vao,
        pData->vbo, pData->ebo, pData->textureId);

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
void* StubPrepareRendererResources::prepareRasterInLoadThread(
    CesiumGltf::ImageAsset& image, const std::any& rendererOptions) {
  return nullptr;
}
void* StubPrepareRendererResources::prepareRasterInMainThread(
    CesiumRasterOverlays::RasterOverlayTile& rasterTile, void* pLoadThreadResult) {
  return nullptr;
}
void StubPrepareRendererResources::freeRaster(
    const CesiumRasterOverlays::RasterOverlayTile& rasterTile, void* pLoadThreadResult,
    void* pMainThreadResult) noexcept {
}
void StubPrepareRendererResources::attachRasterInMainThread(
    const Cesium3DTilesSelection::Tile& tile, int32_t overlayTextureCoordinateID,
    const CesiumRasterOverlays::RasterOverlayTile& rasterTile, void* pMainThreadRendererResources,
    const glm::dvec2& translation, const glm::dvec2& scale) {
}
void StubPrepareRendererResources::detachRasterInMainThread(
    const Cesium3DTilesSelection::Tile& tile, int32_t overlayTextureCoordinateID,
    const CesiumRasterOverlays::RasterOverlayTile& rasterTile,
    void*                                          pMainThreadRendererResources) noexcept {
}

} // namespace csp::cesiumrenderer