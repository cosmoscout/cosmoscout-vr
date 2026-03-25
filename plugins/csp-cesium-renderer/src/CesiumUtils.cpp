////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "CesiumUtils.hpp"
#include "logger.hpp"

// CosmoScout FrameStats for benchmarking timers
#include "../../../src/cs-utils/FrameStats.hpp"
#include <CesiumAsync/AsyncSystem.h>
#include <CesiumGltf/AccessorView.h>
#include <CesiumGltf/Image.h>
#include <CesiumGltf/Material.h>
#include <CesiumGltf/Node.h>
#include <CesiumGltf/Texture.h>
#include <CesiumGltfContent/GltfUtilities.h>
#include <GL/glew.h>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <thread>

namespace csp::cesiumrenderer {

void CosmoScoutTaskProcessor::startTask(std::function<void()> f) {
  std::thread(std::move(f)).detach();
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Helper: Extract a single primitive's data into the render data container.
// nodeTransform is the accumulated local transform for this node (NOT the tile-to-ECEF transform).
// We bake nodeTransform into vertex positions so the flat VBO contains correctly-placed geometry.
////////////////////////////////////////////////////////////////////////////////////////////////////
static void extractPrimitive(CesiumRenderData* renderData, const CesiumGltf::Model* pModel,
    const CesiumGltf::MeshPrimitive& primitive, const glm::dmat4& nodeTransform) {

  // --- Find POSITION accessor ---
  auto posIt = primitive.attributes.find("POSITION");
  if (posIt == primitive.attributes.end()) {
    return;
  }

  CesiumGltf::AccessorView<CesiumGltf::AccessorTypes::VEC3<float>> positions(
      *pModel, posIt->second);
  if (positions.status() != CesiumGltf::AccessorViewStatus::Valid) {
    return;
  }

  // --- Find NORMAL accessor (optional) ---
  bool                                                             hasNormals = false;
  CesiumGltf::AccessorView<CesiumGltf::AccessorTypes::VEC3<float>> normals;
  auto normIt = primitive.attributes.find("NORMAL");
  if (normIt != primitive.attributes.end()) {
    normals =
        CesiumGltf::AccessorView<CesiumGltf::AccessorTypes::VEC3<float>>(*pModel, normIt->second);
    if (normals.status() == CesiumGltf::AccessorViewStatus::Valid) {
      hasNormals = true;
    }
  }

  // --- Find TEXCOORD_0 accessor (optional) ---
  bool                                                             hasUVs = false;
  CesiumGltf::AccessorView<CesiumGltf::AccessorTypes::VEC2<float>> uvs;
  auto uvIt = primitive.attributes.find("TEXCOORD_0");
  if (uvIt != primitive.attributes.end()) {
    uvs = CesiumGltf::AccessorView<CesiumGltf::AccessorTypes::VEC2<float>>(*pModel, uvIt->second);
    if (uvs.status() == CesiumGltf::AccessorViewStatus::Valid) {
      hasUVs = true;
    }
  }

  // --- Get baseColorFactor for this primitive's material ---
  float matR = 0.8f, matG = 0.8f, matB = 0.8f, matA = 1.0f;
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

  // --- Compute normal matrix for this node (inverse transpose of upper-left 3x3) ---
  glm::mat3 normalMatrix = glm::mat3(1.0f);
  if (hasNormals) {
    normalMatrix = glm::mat3(glm::transpose(glm::inverse(glm::mat4(nodeTransform))));
  }

  // --- Copy vertices with node transform baked in ---
  // Layout per vertex: [Px, Py, Pz, Nx, Ny, Nz, U, V, R, G, B, A] = 12 floats
  size_t vertexStart = renderData->vertices.size() / 12;
  renderData->vertices.reserve(
      renderData->vertices.size() + static_cast<size_t>(positions.size()) * 12);

  for (int64_t i = 0; i < positions.size(); ++i) {
    // Transform position by node's local transform
    glm::dvec4 localPos(positions[i].value[0], positions[i].value[1], positions[i].value[2], 1.0);
    glm::dvec4 transformed = nodeTransform * localPos;

    renderData->vertices.push_back(static_cast<float>(transformed.x));
    renderData->vertices.push_back(static_cast<float>(transformed.y));
    renderData->vertices.push_back(static_cast<float>(transformed.z));

    // Transform normal
    if (hasNormals) {
      glm::vec3 n(normals[i].value[0], normals[i].value[1], normals[i].value[2]);
      glm::vec3 tn = glm::normalize(normalMatrix * n);
      renderData->vertices.push_back(tn.x);
      renderData->vertices.push_back(tn.y);
      renderData->vertices.push_back(tn.z);
    } else {
      renderData->vertices.push_back(0.0f);
      renderData->vertices.push_back(0.0f);
      renderData->vertices.push_back(1.0f);
    }

    // UV
    if (hasUVs) {
      renderData->vertices.push_back(uvs[i].value[0]);
      renderData->vertices.push_back(uvs[i].value[1]);
    } else {
      renderData->vertices.push_back(0.0f);
      renderData->vertices.push_back(0.0f);
    }

    // Per-vertex color from material
    renderData->vertices.push_back(matR);
    renderData->vertices.push_back(matG);
    renderData->vertices.push_back(matB);
    renderData->vertices.push_back(matA);
  }

  // --- Extract indices ---
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
            renderData->indices.push_back(
                static_cast<uint32_t>(indexView[i].value[0]) + static_cast<uint32_t>(vertexStart));
          }
        }
      }
    }
  }

  // --- Extract texture (first texture wins across all primitives) ---
  if (!renderData->hasTexture && primitive.material >= 0) {
    const CesiumGltf::Material* pMaterial =
        CesiumGltf::Model::getSafe(&pModel->materials, primitive.material);
    if (pMaterial && pMaterial->pbrMetallicRoughness) {
      const auto& pbr = *pMaterial->pbrMetallicRoughness;
      if (pbr.baseColorTexture) {
        int32_t                    textureIndex = pbr.baseColorTexture->index;
        const CesiumGltf::Texture* pTexture =
            CesiumGltf::Model::getSafe(&pModel->textures, textureIndex);
        if (pTexture && pTexture->source >= 0) {
          const CesiumGltf::Image* pImage =
              CesiumGltf::Model::getSafe(&pModel->images, pTexture->source);
          if (pImage && pImage->pAsset) {
            const CesiumGltf::ImageAsset& asset = *pImage->pAsset;
            if (!asset.pixelData.empty()) {
              renderData->texturePixels = asset.pixelData;
              renderData->texWidth      = asset.width;
              renderData->texHeight     = asset.height;
              renderData->texChannels   = asset.channels;
              renderData->hasTexture    = true;

              logger().info("[Cesium] CPU Thread: Texture extracted: {}x{}, {} channels, {} bytes.",
                  asset.width, asset.height, asset.channels, asset.pixelData.size());
            }
          }
        }
      }
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Helper: Recursively walk the glTF node tree.
// parentTransform accumulates all ancestor transforms.
// This processes each node's mesh (if any) and recurses into children.
////////////////////////////////////////////////////////////////////////////////////////////////////
static void processNode(CesiumRenderData* renderData, const CesiumGltf::Model* pModel,
    int nodeIndex, const glm::dmat4& parentTransform) {

  if (nodeIndex < 0 || nodeIndex >= static_cast<int>(pModel->nodes.size())) {
    return;
  }

  const CesiumGltf::Node& node = pModel->nodes[nodeIndex];

  // Get this node's local transform (uses GltfUtilities for correct TRS decomposition)
  glm::dmat4 localTransform(1.0);
  auto       optTransform = CesiumGltfContent::GltfUtilities::getNodeTransform(node);
  if (optTransform) {
    localTransform = *optTransform;
  }

  // Accumulate: world = parent * local
  glm::dmat4 worldTransform = parentTransform * localTransform;

  // If this node references a mesh, extract all its primitives
  if (node.mesh >= 0 && node.mesh < static_cast<int>(pModel->meshes.size())) {
    const CesiumGltf::Mesh& mesh = pModel->meshes[node.mesh];
    for (const CesiumGltf::MeshPrimitive& primitive : mesh.primitives) {
      extractPrimitive(renderData, pModel, primitive, worldTransform);
    }
  }

  // Recurse into children
  for (int childIndex : node.children) {
    processNode(renderData, pModel, childIndex, worldTransform);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// prepareInLoadThread: CPU-side geometry extraction (runs on background thread)
////////////////////////////////////////////////////////////////////////////////////////////////////

CesiumAsync::Future<Cesium3DTilesSelection::TileLoadResultAndRenderResources>
StubPrepareRendererResources::prepareInLoadThread(const CesiumAsync::AsyncSystem& asyncSystem,
    Cesium3DTilesSelection::TileLoadResult&&                                      tileLoadResult,
    [[maybe_unused]] const glm::dmat4& transform, const std::any& rendererOptions) {

  // --- STEP A: Check if the tile contains a glTF model ---
  CesiumGltf::Model* pModel = std::get_if<CesiumGltf::Model>(&tileLoadResult.contentKind);

  if (!pModel) {
    logger().debug("[Cesium] CPU Thread: Tile has no model data, skipping.");
    return asyncSystem.createResolvedFuture(
        Cesium3DTilesSelection::TileLoadResultAndRenderResources{
            std::move(tileLoadResult), nullptr});
  }

  logger().info("[Cesium] CPU Thread: Model found! Meshes: {}", pModel->meshes.size());

  // ── Benchmarking: measure CPU-side glTF extraction time ──
  // NOTE: FrameStats::ScopedTimer is MAIN-THREAD-ONLY (writes to shared QueryPool).
  // This function runs on a background thread, so we use std::chrono instead.
  auto cpuParseStart = std::chrono::high_resolution_clock::now();

  // --- STEP B: Create our render data container on the heap ---
  auto* renderData = new CesiumRenderData();

  // --- STEP C: Compute CORRECTED root transform ---
  // 'transform' is the tile-to-ECEF matrix from Cesium's selection algorithm.
  // We must apply three corrections that the renderer needs:
  //   1. applyRtcCenter: adds the RTC_CENTER offset (for tiles using relative positioning)
  //   2. applyGltfUpAxisTransform: corrects Y-up → Z-up axis mismatch (per 3D Tiles spec)
  //   3. ECEF → CosmoScout axis permutation (see below)
  glm::dmat4 rootTransform = transform;
  rootTransform = CesiumGltfContent::GltfUtilities::applyRtcCenter(*pModel, rootTransform);
  rootTransform =
      CesiumGltfContent::GltfUtilities::applyGltfUpAxisTransform(*pModel, rootTransform);

  // --- STEP C2: ECEF → CosmoScout-GLM axis permutation ---
  // CosmoScout's core engine swizzles ALL SPICE coordinates when converting to its internal
  // GLM rendering frame (see CelestialAnchor.cpp lines 99 and 110):
  //   GLM.x = SPICE-Y  (90° East longitude)
  //   GLM.y = SPICE-Z  (North Pole)
  //   GLM.z = SPICE-X  (Prime Meridian, 0° longitude)
  //
  // Cesium tiles output standard ECEF where X=Prime Meridian, Y=90°E, Z=North.
  // Without this correction, the Earth mesh appears rotated because the North Pole axis
  // (ECEF-Z) lands on CosmoScout's Z-axis (Prime Meridian) instead of Y-axis (North).
  //
  // The permutation matrix maps: ECEF(X,Y,Z) → GLM(Y,Z,X)
  //   Column 0: ECEF-X (Prime Meridian) → GLM-Z  → (0, 0, 1)
  //   Column 1: ECEF-Y (90° East)       → GLM-X  → (1, 0, 0)
  //   Column 2: ECEF-Z (North Pole)     → GLM-Y  → (0, 1, 0)
  static const glm::dmat4 ecefToCosmoScout(0.0, 0.0, 1.0, 0.0, // col 0: ECEF-X → GLM-Z
      1.0, 0.0, 0.0, 0.0,                                      // col 1: ECEF-Y → GLM-X
      0.0, 1.0, 0.0, 0.0,                                      // col 2: ECEF-Z → GLM-Y
      0.0, 0.0, 0.0, 1.0                                       // col 3: no translation
  );
  rootTransform = ecefToCosmoScout * rootTransform;

  // Store the corrected transform for the renderer to use (64-bit, composed with
  // observerToEarth in the draw loop for observer-relative precision).
  renderData->tileTransform = rootTransform;

  // --- STEP D: Walk the glTF node tree recursively ---
  // Unlike our old code which flat-iterated pModel->meshes (losing all node transforms),
  // we now follow the glTF scene graph: scene → root nodes → children.
  // Per-node transforms (matrix or TRS) are accumulated and baked into vertex positions.
  // The identity matrix is passed as parent because node transforms are LOCAL to the tile.
  glm::dmat4 identity(1.0);

  if (!pModel->scenes.empty()) {
    // Process the default scene (or first scene)
    int sceneIndex = pModel->scene >= 0 ? pModel->scene : 0;
    if (sceneIndex < static_cast<int>(pModel->scenes.size())) {
      const auto& scene = pModel->scenes[sceneIndex];
      for (int rootNodeIndex : scene.nodes) {
        processNode(renderData, pModel, rootNodeIndex, identity);
      }
    }
  } else {
    // No scenes defined — process all root-level nodes (fallback)
    for (int i = 0; i < static_cast<int>(pModel->nodes.size()); ++i) {
      processNode(renderData, pModel, i, identity);
    }
  }

  logger().info("[Cesium] CPU Thread: Extracted {} vertices, {} indices.",
      renderData->vertices.size() / 12, renderData->indices.size());

  // ── Log CPU parsing time for benchmarking ──
  auto cpuParseEnd = std::chrono::high_resolution_clock::now();
  auto cpuParseMs =
      std::chrono::duration_cast<std::chrono::microseconds>(cpuParseEnd - cpuParseStart).count() /
      1000.0;
  logger().info("[Cesium] CPU Parsing took {:.2f} ms.", cpuParseMs);

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

  // ── Benchmarking: measure OpenGL buffer upload time ──
  cs::utils::FrameStats::ScopedTimer timer(
      "Cesium VRAM Upload", cs::utils::FrameStats::TimerMode::eCPU);

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
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
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