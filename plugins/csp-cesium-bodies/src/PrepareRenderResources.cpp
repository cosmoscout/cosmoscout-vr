////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "PrepareRenderResources.hpp"
#include "RenderData.h"
#include "logger.hpp"

#include "../../../src/cs-utils/FrameStats.hpp"
#include <CesiumAsync/AsyncSystem.h>
#include <CesiumGltf/AccessorView.h>
#include <CesiumGltf/Image.h>
#include <CesiumGltf/Material.h>
#include <CesiumGltf/Node.h>
#include <CesiumGltf/Sampler.h>
#include <CesiumGltf/Texture.h>
#include <CesiumGltfContent/GltfUtilities.h>
#include <GL/glew.h>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <string>
#include <thread>

namespace csp::cesiumbodies {

////////////////////////////////////////////////////////////////////////////////////////////////////

void TaskProcessor::startTask(std::function<void()> f) {
  std::thread(std::move(f)).detach();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static int32_t getOrCreateTexture(
    CesiumRenderData* renderData, CesiumGltf::Model const* model, int32_t textureIndex) {
  for (size_t i = 0; i < renderData->textures.size(); ++i) {
    if (renderData->textures[i].sourceIndex == textureIndex) {
      return static_cast<int32_t>(i);
    }
  }

  CesiumGltf::Texture const* texture = CesiumGltf::Model::getSafe(&model->textures, textureIndex);
  if (!texture || texture->source < 0) {
    return -1;
  }

  CesiumGltf::Image const* image = CesiumGltf::Model::getSafe(&model->images, texture->source);
  if (!image || !image->pAsset || image->pAsset->pixelData.empty()) {
    return -1;
  }

  CesiumImage::ImageAsset const& asset = *image->pAsset;

  TextureData textureData;
  textureData.pixels      = asset.pixelData;
  textureData.width       = asset.width;
  textureData.height      = asset.height;
  textureData.channels    = asset.channels;
  textureData.sourceIndex = textureIndex;

  if (auto sampler = CesiumGltf::Model::getSafe(&model->samplers, texture->sampler)) {
    textureData.wrapS = sampler->wrapS;
    textureData.wrapT = sampler->wrapT;
    if (sampler->minFilter) {
      textureData.minFilter = *sampler->minFilter;
    }
    if (sampler->magFilter) {
      textureData.magFilter = *sampler->magFilter;
    }
  }

  renderData->textures.emplace_back(std::move(textureData));
  return static_cast<int32_t>(renderData->textures.size() - 1);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static void extractPrimitive(CesiumRenderData* renderData, CesiumGltf::Model const* model,
    CesiumGltf::MeshPrimitive const& primitive, glm::dmat4 const& nodeTransform) {

  auto positionIt = primitive.attributes.find("POSITION");
  if (positionIt == primitive.attributes.end()) {
    logger().error("Could not find POSITION attribute!");
    return;
  }

  CesiumGltf::AccessorView<CesiumGltf::AccessorTypes::VEC3<float>> positions(
      *model, positionIt->second);
  if (positions.status() != CesiumGltf::AccessorViewStatus::Valid) {
    logger().error("Invalid POSITION accessor!");
    return;
  }

  auto normalIt = primitive.attributes.find("NORMAL");
  if (normalIt == primitive.attributes.end()) {
    logger().error("Could not find NORMAL attribute!");
    return;
  }

  CesiumGltf::AccessorView<CesiumGltf::AccessorTypes::VEC3<float>> normals(
      *model, normalIt->second);
  if (normals.status() != CesiumGltf::AccessorViewStatus::Valid) {
    logger().error("Invalid NORMAL accessor!");
    return;
  }

  const CesiumGltf::Material* material =
      CesiumGltf::Model::getSafe(&model->materials, primitive.material);
  const CesiumGltf::MaterialPBRMetallicRoughness* pbr =
      material && material->pbrMetallicRoughness ? &*material->pbrMetallicRoughness : nullptr;

  int64_t texCoordSet = pbr && pbr->baseColorTexture ? pbr->baseColorTexture->texCoord : 0;
  bool    hasUVs      = false;
  CesiumGltf::AccessorView<CesiumGltf::AccessorTypes::VEC2<float>> uvs;
  if (auto uvIt = primitive.attributes.find("TEXCOORD_" + std::to_string(texCoordSet));
      uvIt != primitive.attributes.end()) {
    uvs = CesiumGltf::AccessorView<CesiumGltf::AccessorTypes::VEC2<float>>(*model, uvIt->second);
    if (uvs.status() == CesiumGltf::AccessorViewStatus::Valid) {
      hasUVs = true;
    }
  }

  glm::vec4 color{1.0F};
  if (pbr) {
    if (const auto& factor = pbr->baseColorFactor; factor.size() >= 4) {
      color.x = static_cast<float>(factor[0]);
      color.y = static_cast<float>(factor[1]);
      color.z = static_cast<float>(factor[2]);
      color.w = static_cast<float>(factor[3]);
    }
  }

  size_t vertexStart = renderData->vertices.size() / 12;
  renderData->vertices.reserve(
      renderData->vertices.size() + static_cast<size_t>(positions.size()) * 12);

  for (int64_t i = 0; i < positions.size(); ++i) {
    glm::dvec4 localPos(positions[i].value[0], positions[i].value[1], positions[i].value[2], 1.0);
    glm::dvec4 transformed = nodeTransform * localPos;

    renderData->vertices.push_back(static_cast<float>(transformed.x));
    renderData->vertices.push_back(static_cast<float>(transformed.y));
    renderData->vertices.push_back(static_cast<float>(transformed.z));

    renderData->cpuPositions.emplace_back(transformed.x, transformed.y, transformed.z);

    renderData->vertices.push_back(normals[i].value[0]);
    renderData->vertices.push_back(normals[i].value[1]);
    renderData->vertices.push_back(normals[i].value[2]);

    if (hasUVs) {
      renderData->vertices.push_back(uvs[i].value[0]);
      renderData->vertices.push_back(uvs[i].value[1]);
    } else {
      renderData->vertices.push_back(0.0f);
      renderData->vertices.push_back(0.0f);
    }

    renderData->vertices.push_back(color.x);
    renderData->vertices.push_back(color.y);
    renderData->vertices.push_back(color.z);
    renderData->vertices.push_back(color.w);
  }

  const auto firstIndex = static_cast<uint32_t>(renderData->indices.size());
  if (primitive.indices >= 0) {
    const CesiumGltf::Accessor* indexAccessor =
        CesiumGltf::Model::getSafe(&model->accessors, primitive.indices);

    if (indexAccessor) {
      if (indexAccessor->componentType == CesiumGltf::Accessor::ComponentType::UNSIGNED_SHORT) {
        CesiumGltf::AccessorView<CesiumGltf::AccessorTypes::SCALAR<uint16_t>> indexView(
            *model, primitive.indices);
        if (indexView.status() == CesiumGltf::AccessorViewStatus::Valid) {
          for (int64_t i = 0; i < indexView.size(); ++i) {
            renderData->indices.push_back(
                static_cast<uint32_t>(indexView[i].value[0]) + static_cast<uint32_t>(vertexStart));
          }
        }
      } else if (indexAccessor->componentType ==
                 CesiumGltf::Accessor::ComponentType::UNSIGNED_INT) {
        CesiumGltf::AccessorView<CesiumGltf::AccessorTypes::SCALAR<uint32_t>> indexView(
            *model, primitive.indices);
        if (indexView.status() == CesiumGltf::AccessorViewStatus::Valid) {
          for (int64_t i = 0; i < indexView.size(); ++i) {
            renderData->indices.push_back(
                indexView[i].value[0] + static_cast<uint32_t>(vertexStart));
          }
        }
      } else if (indexAccessor->componentType ==
                 CesiumGltf::Accessor::ComponentType::UNSIGNED_BYTE) {
        CesiumGltf::AccessorView<CesiumGltf::AccessorTypes::SCALAR<uint8_t>> indexView(
            *model, primitive.indices);
        if (indexView.status() == CesiumGltf::AccessorViewStatus::Valid) {
          for (int64_t i = 0; i < indexView.size(); ++i) {
            renderData->indices.push_back(
                static_cast<uint32_t>(indexView[i].value[0]) + static_cast<uint32_t>(vertexStart));
          }
        }
      }
    }
  }

  if (const uint32_t indexCount = static_cast<uint32_t>(renderData->indices.size()) - firstIndex;
      indexCount > 0) {
    int32_t textureSlot = -1;
    if (hasUVs && pbr && pbr->baseColorTexture) {
      textureSlot = getOrCreateTexture(renderData, model, pbr->baseColorTexture->index);
    }
    renderData->batches.push_back(
        {.firstIndex = firstIndex, .indexCount = indexCount, .textureSlot = textureSlot});
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static void processNode(CesiumRenderData* renderData, CesiumGltf::Model const* model, int nodeIndex,
    glm::dmat4 const& parentTransform) {

  if (nodeIndex < 0 || nodeIndex >= static_cast<int>(model->nodes.size())) {
    return;
  }

  CesiumGltf::Node const& node = model->nodes[nodeIndex];

  // Get this node's local transform (uses GltfUtilities for correct TRS decomposition)
  glm::dmat4 localTransform(1.0);
  if (auto optTransform = CesiumGltfContent::GltfUtilities::getNodeTransform(node)) {
    localTransform = *optTransform;
  }

  glm::dmat4 worldTransform = parentTransform * localTransform;

  if (node.mesh >= 0 && node.mesh < static_cast<int>(model->meshes.size())) {
    for (const CesiumGltf::Mesh&          mesh = model->meshes[node.mesh];
         const CesiumGltf::MeshPrimitive& primitive : mesh.primitives) {
      extractPrimitive(renderData, model, primitive, worldTransform);
    }
  }

  for (int childIndex : node.children) {
    processNode(renderData, model, childIndex, worldTransform);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Rebase all tile-local vertices around a double-precision anchor before uploading the VBO.
///
/// glTF POSITION accessors are floats. That is normally sufficient for a tile-sized offset, but
/// some tiles contain coordinates that are still large even after RTC_CENTER has been applied.
/// Converting those coordinates directly to a float VBO quantizes neighboring vertices together.
/// Moving the anchor into the tile transform keeps the GPU coordinates small while preserving the
/// exact placement of the geometry.
static void rebaseVertices(CesiumRenderData* renderData) {
  if (renderData->cpuPositions.empty()) {
    return;
  }

  const glm::dvec3 origin = renderData->cpuPositions.front();

  // The CPU copy must use the same rebased coordinate system as the VBO because intersection
  // transforms rays through tileTransform.
  for (glm::dvec3& position : renderData->cpuPositions) {
    position -= origin;
  }

  // Apply the inverse change to the root transform: T * (origin + local) = (T * To) * local.
  glm::dmat4 originTransform(1.0);
  originTransform[3] = glm::dvec4(origin, 1.0);
  renderData->tileTransform *= originTransform;

  // Replace only the position part of the interleaved VBO. The remaining attributes are already
  // finalized by extractPrimitive().
  for (size_t i = 0; i < renderData->cpuPositions.size(); ++i) {
    const size_t vertexOffset = i * 12;
    if (vertexOffset + 2 >= renderData->vertices.size()) {
      break;
    }

    renderData->vertices[vertexOffset + 0] = static_cast<float>(renderData->cpuPositions[i].x);
    renderData->vertices[vertexOffset + 1] = static_cast<float>(renderData->cpuPositions[i].y);
    renderData->vertices[vertexOffset + 2] = static_cast<float>(renderData->cpuPositions[i].z);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

CesiumAsync::Future<Cesium3DTilesSelection::TileLoadResultAndRenderResources>
PrepareRendererResources::prepareInLoadThread(const CesiumAsync::AsyncSystem& asyncSystem,
    Cesium3DTilesSelection::TileLoadResult&&                                  tileLoadResult,
    [[maybe_unused]] const glm::dmat4& transform, const std::any& rendererOptions) {

  auto* model = std::get_if<CesiumGltf::Model>(&tileLoadResult.contentKind);

  if (!model) {
    return asyncSystem.createResolvedFuture(
        Cesium3DTilesSelection::TileLoadResultAndRenderResources{
            .result = std::move(tileLoadResult), .pRenderResources = nullptr});
  }

  auto* renderData = new CesiumRenderData();

  glm::dmat4 rootTransform = transform;
  rootTransform = CesiumGltfContent::GltfUtilities::applyRtcCenter(*model, rootTransform);
  rootTransform =
      CesiumGltfContent::GltfUtilities::applyGltfUpAxisTransform(*model, rootTransform);

  static constexpr glm::dmat4 ecefToCosmoScout(0.0, 0.0, 1.0, 0.0, // col 0: ECEF-X → GLM-Z
      1.0, 0.0, 0.0, 0.0,                                          // col 1: ECEF-Y → GLM-X
      0.0, 1.0, 0.0, 0.0,                                          // col 2: ECEF-Z → GLM-Y
      0.0, 0.0, 0.0, 1.0                                           // col 3: no translation
  );

  rootTransform             = ecefToCosmoScout * rootTransform;
  renderData->tileTransform = rootTransform;
  glm::dmat4 identity(1.0);

  if (!model->scenes.empty()) {
    if (int sceneIndex = model->scene >= 0 ? model->scene : 0;
        sceneIndex < static_cast<int>(model->scenes.size())) {
      for (const auto& scene = model->scenes[sceneIndex]; int rootNodeIndex : scene.nodes) {
        processNode(renderData, model, rootNodeIndex, identity);
      }
    }
  } else {
    for (int i = 0; i < static_cast<int>(model->nodes.size()); ++i) {
      processNode(renderData, model, i, identity);
    }
  }

  rebaseVertices(renderData);

  return asyncSystem.createResolvedFuture(Cesium3DTilesSelection::TileLoadResultAndRenderResources{
      std::move(tileLoadResult), renderData});
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void* PrepareRendererResources::prepareInMainThread(
    Cesium3DTilesSelection::Tile& tile, void* pLoadThreadResult) {

  auto* data = static_cast<CesiumRenderData*>(pLoadThreadResult);

  if (!data || data->vertices.empty()) {
    return nullptr;
  }

  cs::utils::FrameStats::ScopedTimer timer("Cesium VRAM Upload");

  glGenVertexArrays(1, &data->vao);
  glBindVertexArray(data->vao);

  glGenBuffers(1, &data->vbo);
  glBindBuffer(GL_ARRAY_BUFFER, data->vbo);
  glBufferData(GL_ARRAY_BUFFER, data->vertices.size() * sizeof(float), data->vertices.data(),
      GL_STATIC_DRAW);

  glGenBuffers(1, &data->ebo);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, data->ebo);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, data->indices.size() * sizeof(uint32_t),
      data->indices.data(), GL_STATIC_DRAW);

  GLsizei stride = 12 * sizeof(float);

  // Location 0 = Position (3 floats, offset 0 bytes)
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, nullptr);
  glEnableVertexAttribArray(0);

  // Location 1 = Normal (3 floats, offset 12 bytes)
  glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, reinterpret_cast<void*>(3 * sizeof(float)));
  glEnableVertexAttribArray(1);

  // Location 2 = UV (2 floats, offset 24 bytes)
  glVertexAttribPointer(
      2, 2, GL_FLOAT, GL_FALSE, stride, reinterpret_cast<void*>(6 * sizeof(float)));
  glEnableVertexAttribArray(2);

  // Location 3 = Color (4 floats, offset 32 bytes)
  glVertexAttribPointer(
      3, 4, GL_FLOAT, GL_FALSE, stride, reinterpret_cast<void*>(8 * sizeof(float)));
  glEnableVertexAttribArray(3);

  glBindVertexArray(0);

  for (TextureData& texture : data->textures) {
    if (texture.pixels.empty()) {
      continue;
    }

    GLenum format = GL_RGBA;
    if (texture.channels == 1)
      format = GL_RED;
    else if (texture.channels == 2)
      format = GL_RG;
    else if (texture.channels == 3)
      format = GL_RGB;

    glGenTextures(1, &texture.textureId);
    glBindTexture(GL_TEXTURE_2D, texture.textureId);

    // Cesium pixels are tightly packed — no row padding
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

    glTexImage2D(GL_TEXTURE_2D, // target
        0,                      // mip level 0 (the full-size base image)
        format,                 // internal format (how GPU stores it)
        texture.width,          // width in pixels
        texture.height,         // height in pixels
        0,                      // border (always 0, legacy parameter)
        format,                 // pixel data format (how OUR bytes are laid out)
        GL_UNSIGNED_BYTE,       // each channel is one byte (0-255)
        texture.pixels.data()   // pointer to the raw bytes
    );

    // Restore OpenGL default alignment to avoid contaminating other code
    glPixelStorei(GL_UNPACK_ALIGNMENT, 4);

    glGenerateMipmap(GL_TEXTURE_2D);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, texture.wrapS);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, texture.wrapT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, texture.minFilter);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, texture.magFilter);

    glBindTexture(GL_TEXTURE_2D, 0);

    texture.pixels.clear();
    texture.pixels.shrink_to_fit();
  }

  data->vertices.clear();
  data->vertices.shrink_to_fit();
  data->indices.clear();
  data->indices.shrink_to_fit();

  return data;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void PrepareRendererResources::free(
    Cesium3DTilesSelection::Tile& tile, void* loadThreadResult, void* pMainThreadResult) noexcept {

  if (pMainThreadResult) {
    auto* data = static_cast<CesiumRenderData*>(pMainThreadResult);

    // Delete the GPU resources from VRAM
    if (data->vao != 0) {
      glDeleteVertexArrays(1, &data->vao);
    }
    if (data->vbo != 0) {
      glDeleteBuffers(1, &data->vbo);
    }
    if (data->ebo != 0) {
      glDeleteBuffers(1, &data->ebo);
    }
    for (const TextureData& texture : data->textures) {
      if (texture.textureId != 0) {
        glDeleteTextures(1, &texture.textureId);
      }
    }

    delete data;
  }

  if (loadThreadResult) {
    auto* pData = static_cast<CesiumRenderData*>(loadThreadResult);
    delete pData;
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void* PrepareRendererResources::prepareRasterInLoadThread(
    CesiumImage::ImageAsset& image, const std::any& rendererOptions) {
  return nullptr;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void* PrepareRendererResources::prepareRasterInMainThread(
    CesiumRasterOverlays::RasterOverlayTile& rasterTile, void* pLoadThreadResult) {
  return nullptr;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void PrepareRendererResources::freeRaster(const CesiumRasterOverlays::RasterOverlayTile& rasterTile,
    void* pLoadThreadResult, void* pMainThreadResult) noexcept {
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void PrepareRendererResources::attachRasterInMainThread(const Cesium3DTilesSelection::Tile& tile,
    int32_t overlayTextureCoordinateID, const CesiumRasterOverlays::RasterOverlayTile& rasterTile,
    void* pMainThreadRendererResources, const glm::dvec2& translation, const glm::dvec2& scale) {
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void PrepareRendererResources::detachRasterInMainThread(const Cesium3DTilesSelection::Tile& tile,
    int32_t overlayTextureCoordinateID, const CesiumRasterOverlays::RasterOverlayTile& rasterTile,
    void* pMainThreadRendererResources) noexcept {
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::cesiumbodies
