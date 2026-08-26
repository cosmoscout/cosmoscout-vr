////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "TilesetRenderer.hpp"
#include "RenderData.h"
#include "logger.hpp"

#include "../../../src/cs-core/SolarSystem.hpp"
#include "../../../src/cs-utils/FrameStats.hpp"
#include "../../../src/cs-utils/convert.hpp"
#include "../../../src/cs-utils/utils.hpp"

#include <VistaKernel/GraphicsManager/VistaGroupNode.h>
#include <VistaKernel/GraphicsManager/VistaSceneGraph.h>
#include <VistaKernel/VistaSystem.h>
#include <VistaKernelOpenSGExt/VistaOpenSGMaterialTools.h>

#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <Cesium3DTilesSelection/Tile.h>
#include <Cesium3DTilesSelection/TileContent.h>
#include <Cesium3DTilesSelection/TilesetViewGroup.h>
#include <CesiumGeometry/Ray.h>
#include <CesiumGltfContent/GltfUtilities.h>

namespace csp::cesiumbodies {

const char* TilesetRenderer::CESIUM_VERT = R"(
#version 430

uniform mat4 uModelMatrix;
uniform mat4 uViewMatrix;
uniform mat4 uProjectionMatrix;

layout(location = 0) in vec3 aPosition;
layout(location = 1) in vec2 aUV;
layout(location = 2) in vec4 aColor;

out vec2 vUV;
out vec4 vColor;

void main() {
  vec4 viewPos = uViewMatrix * uModelMatrix * vec4(aPosition, 1.0);
  vUV         = aUV;
  vColor      = aColor;
  gl_Position  = uProjectionMatrix * viewPos;
}
)";

////////////////////////////////////////////////////////////////////////////////////////////////////

const char* TilesetRenderer::CESIUM_FRAG = R"(
#version 430

in vec2 vUV;
in vec4 vColor;

uniform sampler2D uBaseColorTexture;
uniform bool      uHasTexture;
uniform float     uSunIlluminance;

layout(location = 0) out vec3 oColor;

const float PI = 3.14159265359;

vec3 sRGBtoLinear(vec3 srgb) {
    return pow(srgb, vec3(2.2));
}

void main() {
  vec3 baseColor;
  if (uHasTexture) {
      baseColor = sRGBtoLinear(texture(uBaseColorTexture, vUV).rgb) * vColor.rgb;
  } else {
      baseColor = vColor.rgb;
  }

  oColor = baseColor * uSunIlluminance * 0.06 / PI;
}
)";

static GLuint compileShader(GLenum type, const char* source) { //
  GLuint shader = glCreateShader(type);
  glShaderSource(shader, 1, &source, nullptr);
  glCompileShader(shader);

  GLint success = 0;
  glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
  if (!success) {
    char infoLog[512];
    glGetShaderInfoLog(shader, 512, nullptr, infoLog);
    logger().error("Shader compilation failed: {}", infoLog);
    glDeleteShader(shader);
    return 0;
  }
  return shader;
}

TilesetRenderer::TilesetRenderer(
    Cesium3DTilesSelection::Tileset* pTileset, std::shared_ptr<cs::core::SolarSystem> pSolarSystem)
    : mTileset(pTileset)
    , mSolarSystem(std::move(pSolarSystem)) {

  GLuint vert = compileShader(GL_VERTEX_SHADER, CESIUM_VERT);
  GLuint frag = compileShader(GL_FRAGMENT_SHADER, CESIUM_FRAG);

  if (vert && frag) {
    mShaderProgram = glCreateProgram();
    glAttachShader(mShaderProgram, vert);
    glAttachShader(mShaderProgram, frag);
    glLinkProgram(mShaderProgram);

    GLint linked = 0;
    glGetProgramiv(mShaderProgram, GL_LINK_STATUS, &linked);
    if (!linked) {
      char infoLog[512];
      glGetProgramInfoLog(mShaderProgram, 512, nullptr, infoLog);
      logger().error("Shader link failed: {}", infoLog);
      glDeleteProgram(mShaderProgram);
      mShaderProgram = 0;
    } else {
      mLocModelMatrix      = glGetUniformLocation(mShaderProgram, "uModelMatrix");
      mLocViewMatrix       = glGetUniformLocation(mShaderProgram, "uViewMatrix");
      mLocProjectionMatrix = glGetUniformLocation(mShaderProgram, "uProjectionMatrix");
      mLocBaseColorTexture = glGetUniformLocation(mShaderProgram, "uBaseColorTexture");
      mLocHasTexture       = glGetUniformLocation(mShaderProgram, "uHasTexture");
      mLocSunIlluminance   = glGetUniformLocation(mShaderProgram, "uSunIlluminance");
    }
  }

  glDeleteShader(vert);
  glDeleteShader(frag);

  VistaSceneGraph* pSG = GetVistaSystem()->GetGraphicsManager()->GetSceneGraph();
  mGLNode.reset(pSG->NewOpenGLNode(pSG->GetRoot(), this));
  VistaOpenSGMaterialTools::SetSortKeyOnSubtree(
      mGLNode.get(), static_cast<int>(cs::utils::DrawOrder::eOpaqueItems));

  logger().info("CesiumTilesetRenderer attached to ViSTA scene graph.");
}

bool TilesetRenderer::Do() {
  // TODO: Extract into separate body.
  auto earth = mSolarSystem->getObject("Earth");
  if (mShaderProgram == 0 || !earth) {
    return true;
  }

  cs::utils::FrameStats::ScopedTimer timer("Cesium Tileset Rendering");

  static int frameCounter = 0;
  frameCounter++;

  glUseProgram(mShaderProgram);

  std::array<GLfloat, 16> glMatV{};
  std::array<GLfloat, 16> glMatP{};
  glGetFloatv(GL_MODELVIEW_MATRIX, glMatV.data());
  glGetFloatv(GL_PROJECTION_MATRIX, glMatP.data());

  glUniformMatrix4fv(mLocViewMatrix, 1, GL_FALSE, glMatV.data());
  glUniformMatrix4fv(mLocProjectionMatrix, 1, GL_FALSE, glMatP.data());

  glm::dmat4 observerToEarth = earth->getObserverRelativeTransform();
  glm::dvec3 earthPos(observerToEarth[3]);

  auto sunIlluminance = static_cast<float>(mSolarSystem->getSunIlluminance(earthPos));
  glUniform1f(mLocSunIlluminance, sunIlluminance);

  GLint     prevDepthFunc;
  GLboolean cullEnabled  = glIsEnabled(GL_CULL_FACE);
  GLboolean blendEnabled = glIsEnabled(GL_BLEND);

  glGetIntegerv(GL_DEPTH_FUNC, &prevDepthFunc);

  glDepthFunc(GL_GEQUAL);
  glDisable(GL_CULL_FACE);
  glDisable(GL_BLEND);

  const auto& result = mTileset->getDefaultViewGroup().getViewUpdateResult();
  const auto& tiles  = result.tilesToRenderThisFrame;

  uint32_t tilesDrawn = 0;

  for (auto const& pTilePointer : tiles) {
    const auto* pTile = pTilePointer.get();

    if (auto state = pTile->getState();
        state != Cesium3DTilesSelection::TileLoadState::ContentLoaded &&
        state != Cesium3DTilesSelection::TileLoadState::Done) {
      continue;
    }

    auto* pRenderContent = pTile->getContent().getRenderContent();
    if (!pRenderContent)
      continue;

    auto* pData = static_cast<CesiumRenderData*>(pRenderContent->getRenderResources());
    if (!pData || pData->vao == 0)
      continue;

    glm::dmat4 tileToObserver = observerToEarth * pData->tileTransform;
    glm::mat4  modelMatrix(tileToObserver);

    glUniformMatrix4fv(mLocModelMatrix, 1, GL_FALSE, glm::value_ptr(modelMatrix));

    {
      cs::utils::FrameStats::ScopedTimer drawTimer("Cesium GPU Draw");
      glBindVertexArray(pData->vao);
      glActiveTexture(GL_TEXTURE0);
      glUniform1i(mLocBaseColorTexture, 0);

      for (const DrawBatch& batch : pData->batches) {
        GLuint textureId = 0;
        if (batch.textureSlot >= 0 &&
            batch.textureSlot < static_cast<int32_t>(pData->textures.size())) {
          textureId = pData->textures[batch.textureSlot].textureId;
        }

        glBindTexture(GL_TEXTURE_2D, textureId);
        glUniform1i(mLocHasTexture, textureId != 0 ? 1 : 0);
        glDrawElements(GL_TRIANGLES, static_cast<GLsizei>(batch.indexCount), GL_UNSIGNED_INT,
            reinterpret_cast<void*>(static_cast<uintptr_t>(batch.firstIndex) * sizeof(uint32_t)));
      }
    }

    tilesDrawn++;
  }

  glDepthFunc(prevDepthFunc);
  if (cullEnabled)
    glEnable(GL_CULL_FACE);
  if (blendEnabled)
    glEnable(GL_BLEND);
  glBindVertexArray(0);
  glUseProgram(0);

  return true;
}

// Cesium native only offers an async function to test for intersections. We query this intersection
// asynchronously, which works quite well in most cases. The only downside is that collisions with
// the ground are a little bouncy.
double TilesetRenderer::getHeight(glm::dvec2 lngLat) const {
  if (!mHeightQueryInFlight) {
    mHeightQueryInFlight = true;
    mLastQueryLngLat     = lngLat;

    std::vector positions{CesiumGeospatial::Cartographic(lngLat.x, lngLat.y, 0.0)};

    mTileset->sampleHeightMostDetailed(positions).thenInMainThread(
        [this](Cesium3DTilesSelection::SampleHeightResult&& result) {
          mHeightQueryInFlight = false;
          if (!result.positions.empty() && result.sampleSuccess[0]) {
            mCachedHeight = result.positions[0].height;
          }
        });
  }

  return mCachedHeight;
}

bool TilesetRenderer::getIntersection(
    glm::dvec3 const& rayPos, glm::dvec3 const& rayDir, glm::dvec3& pos) const {
  if (mTileset == nullptr) {
    return false;
  }

  const double dirLength = glm::length(rayDir);
  if (dirLength <= 0.0) {
    return false;
  }

  // CosmoScout VR uses (Z, X, Y) layout, Cesium uses (X, Y, Z) ECEF.
  // Convert ray from CosmoScout coordinates to ECEF.
  glm::dvec3 rayPosECEF(rayPos.yzx());
  glm::dvec3 rayDirECEF(rayDir.yzx());

  // CesiumGeometry::Ray requires a normalized direction.
  const CesiumGeometry::Ray ray(rayPosECEF, rayDirECEF / dirLength);

  bool       found         = false;
  double     closestDistSq = std::numeric_limits<double>::max();
  glm::dvec3 closestPointECEF(0.0);

  mTileset->forEachLoadedTile([&](Cesium3DTilesSelection::Tile const& tile) {
    Cesium3DTilesSelection::TileRenderContent const* pRenderContent =
        tile.getContent().getRenderContent();
    if (pRenderContent == nullptr) {
      return;
    }

    CesiumGltf::Model const& model = pRenderContent->getModel();

    CesiumGltfContent::GltfUtilities::IntersectResult result =
        CesiumGltfContent::GltfUtilities::intersectRayGltfModel(ray, model,
            /* cullBackFaces */ true, tile.getTransform());

    if (result.hit.has_value()) {
      const double distSq = result.hit->rayToWorldPointDistanceSq;
      if (distSq < closestDistSq) {
        closestDistSq    = distSq;
        closestPointECEF = result.hit->worldPoint;
        found            = true;
      }
    }
  });

  if (found) {
    // Convert intersection point back from ECEF to CosmoScout coordinates.
    pos = glm::dvec3(closestPointECEF.yzx());
  }
  return found;
}

TilesetRenderer::~TilesetRenderer() {
  if (mShaderProgram) {
    glDeleteProgram(mShaderProgram);
  }

  VistaSceneGraph* pSG = GetVistaSystem()->GetGraphicsManager()->GetSceneGraph();
  pSG->GetRoot()->DisconnectChild(mGLNode.get());
}

bool TilesetRenderer::GetBoundingBox(VistaBoundingBox& /*bb*/) {
  return false;
}

} // namespace csp::cesiumbodies
