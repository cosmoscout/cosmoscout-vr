////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "CesiumTilesetRenderer.hpp"
#include "CesiumUtils.hpp"
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

namespace csp::cesiumbodies {

const char* CesiumTilesetRenderer::CESIUM_VERT = R"(
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

const char* CesiumTilesetRenderer::CESIUM_FRAG = R"(
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
      baseColor = sRGBtoLinear(texture(uBaseColorTexture, vUV).rgb);
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

CesiumTilesetRenderer::CesiumTilesetRenderer(
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

bool CesiumTilesetRenderer::Do() {
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

    if (pData->textureId != 0) {
      glActiveTexture(GL_TEXTURE0);
      glBindTexture(GL_TEXTURE_2D, pData->textureId);
      glUniform1i(mLocBaseColorTexture, 0);
      glUniform1i(mLocHasTexture, 1);
    } else {
      glUniform1i(mLocHasTexture, 0);
    }

    {
      cs::utils::FrameStats::ScopedTimer drawTimer("Cesium GPU Draw");
      glBindVertexArray(pData->vao);
      glDrawElements(
          GL_TRIANGLES, static_cast<GLsizei>(pData->indexCount), GL_UNSIGNED_INT, nullptr);
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

/// Tests whether a ray hits a triangle. Returns true if hit, and sets 'tOut' to the parametric
/// distance along the ray (hit point = origin + tOut * direction).
static bool rayTriangleIntersect(glm::dvec3 const& origin, glm::dvec3 const& dir,
    glm::dvec3 const& v0, glm::dvec3 const& v1, glm::dvec3 const& v2, double& tOut) {

  constexpr double EPSILON = 1e-9;

  glm::dvec3 e1 = v1 - v0;
  glm::dvec3 e2 = v2 - v0;
  glm::dvec3 h  = glm::cross(dir, e2);
  double     a  = glm::dot(e1, h);

  if (a > -EPSILON && a < EPSILON) {
    return false;
  }

  double     f = 1.0 / a;
  glm::dvec3 s = origin - v0;
  double     u = f * glm::dot(s, h);

  if (u < 0.0 || u > 1.0) {
    return false;
  }

  glm::dvec3 q = glm::cross(s, e1);
  if (double v = f * glm::dot(dir, q); v < 0.0 || u + v > 1.0) {
    return false;
  }

  if (double t = f * glm::dot(e2, q); t > EPSILON) {
    tOut = t;
    return true;
  }

  return false;
}

// TODO
double CesiumTilesetRenderer::getHeight(glm::dvec2 lngLat) const {
  return 0.0;
}

bool CesiumTilesetRenderer::getIntersection(
    glm::dvec3 const& rayPos, glm::dvec3 const& rayDir, glm::dvec3& pos) const {
  if (!mTileset) {
    return false;
  }

  auto earth = mSolarSystem->getObject("Earth");
  if (!earth) {
    return false;
  }

  const auto& result = mTileset->getDefaultViewGroup().getViewUpdateResult();
  const auto& tiles  = result.tilesToRenderThisFrame;

  double closestT = std::numeric_limits<double>::max();
  bool   foundHit = false;

  for (auto const& pTilePointer : tiles) {
    const auto* pTile = pTilePointer.get();

    auto state = pTile->getState();
    if (state != Cesium3DTilesSelection::TileLoadState::ContentLoaded &&
        state != Cesium3DTilesSelection::TileLoadState::Done) {
      continue;
    }

    auto* pRenderContent = pTile->getContent().getRenderContent();
    if (!pRenderContent) {
      continue;
    }

    auto* pData = static_cast<CesiumRenderData*>(pRenderContent->getRenderResources());
    if (!pData || pData->cpuPositions.empty() || pData->cpuIndices.size() < 3) {
      continue;
    }

    glm::dmat4 tileXform    = pData->tileTransform;
    glm::dmat4 invTileXform = glm::inverse(tileXform);

    glm::dvec3 localOrigin(invTileXform * glm::dvec4(rayPos, 1.0));
    glm::dvec3 localDir = glm::normalize(glm::dvec3(invTileXform * glm::dvec4(rayDir, 0.0)));

    for (size_t i = 0; i + 2 < pData->cpuIndices.size(); i += 3) {
      uint32_t i0 = pData->cpuIndices[i + 0];
      uint32_t i1 = pData->cpuIndices[i + 1];
      uint32_t i2 = pData->cpuIndices[i + 2];

      if (i0 >= pData->cpuPositions.size() || i1 >= pData->cpuPositions.size() ||
          i2 >= pData->cpuPositions.size()) {
        continue;
      }

      glm::dvec3 v0(pData->cpuPositions[i0]);
      glm::dvec3 v1(pData->cpuPositions[i1]);
      glm::dvec3 v2(pData->cpuPositions[i2]);

      if (double t = 0.0; rayTriangleIntersect(localOrigin, localDir, v0, v1, v2, t)) {
        if (t > 0.0 && t < closestT) {
          closestT = t;

          glm::dvec3 localHit = localOrigin + t * localDir;
          pos                 = glm::dvec3(tileXform * glm::dvec4(localHit, 1.0));
          foundHit            = true;
        }
      }
    }
  }

  return foundHit;
}

CesiumTilesetRenderer::~CesiumTilesetRenderer() {
  if (mShaderProgram) {
    glDeleteProgram(mShaderProgram);
  }

  VistaSceneGraph* pSG = GetVistaSystem()->GetGraphicsManager()->GetSceneGraph();
  pSG->GetRoot()->DisconnectChild(mGLNode.get());
}

bool CesiumTilesetRenderer::GetBoundingBox(VistaBoundingBox& /*bb*/) {
  return false;
}

} // namespace csp::cesiumbodies
