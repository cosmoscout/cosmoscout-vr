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

#include <glm/glm.hpp>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <Cesium3DTilesSelection/Tile.h>
#include <Cesium3DTilesSelection/TileContent.h>
#include <Cesium3DTilesSelection/TilesetViewGroup.h>
#include <CesiumGeometry/Ray.h>
#include <CesiumGltfContent/GltfUtilities.h>

namespace csp::cesiumbodies {

////////////////////////////////////////////////////////////////////////////////////////////////////

const char* TilesetRenderer::CESIUM_VERT = R"(
uniform mat4 uModelMatrix;
uniform mat4 uViewMatrix;
uniform mat4 uProjectionMatrix;
uniform vec3 uSunDirection;

layout(location = 0) in vec3 aPosition;
layout(location = 1) in vec2 aUV;
layout(location = 2) in vec4 aColor;

out vec2 vUV;
out vec4 vColor;
out vec3 vSunDirection;
out vec3 vPosition;

void main() {
  vec4 worldPos = uModelMatrix * vec4(aPosition, 1.0);
  vec4 viewPos  = uViewMatrix * worldPos;
  vUV           = aUV;
  vColor        = aColor;
  vSunDirection = (uModelMatrix * vec4(uSunDirection, 0.0)).xyz;
  vPosition     = worldPos.xyz;
  gl_Position   = uProjectionMatrix * viewPos;
}
)";

////////////////////////////////////////////////////////////////////////////////////////////////////

const char* TilesetRenderer::CESIUM_FRAG = R"(
in vec2 vUV;
in vec4 vColor;
in vec3 vSunDirection;
in vec3 vPosition;

uniform sampler2D uBaseColorTexture;
uniform bool      uHasTexture;
uniform float     uSunIlluminance;
uniform float     uAmbientBrightness;
uniform bool      uEnableLighting;
uniform float     uAvgLinearImgIntensity;

layout(location = 0) out vec3 oColor;

const float PI = 3.14159265359;
const float E  = 2.718281828;

vec3 SRGBtoLINEAR(vec3 srgbIn) {
  vec3 bLess = step(vec3(0.04045), srgbIn);
  return mix(srgbIn / vec3(12.92), pow((srgbIn + vec3(0.055)) / vec3(1.055), vec3(2.4)), bLess);
}

void main() {
  vec3 baseColor;
  if (uHasTexture) {
      baseColor = texture(uBaseColorTexture, vUV).rgb * vColor.rgb;
  } else {
      baseColor = vColor.rgb;
  }

#ifdef ENABLE_HDR
  // Make the amount of ambient brightness perceptually linear in HDR mode.
  float ambient = pow(uAmbientBrightness, E);
  baseColor = SRGBtoLINEAR(baseColor) * uSunIlluminance / uAvgLinearImgIntensity;
  baseColor /= PI;
#endif

  oColor = baseColor;
}
)";

////////////////////////////////////////////////////////////////////////////////////////////////////

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

////////////////////////////////////////////////////////////////////////////////////////////////////

TilesetRenderer::TilesetRenderer(Cesium3DTilesSelection::Tileset* pTileset,
    std::shared_ptr<const cs::scene::CelestialObject>             object,
    std::shared_ptr<cs::core::SolarSystem>                        pSolarSystem,
    std::shared_ptr<cs::core::Settings> settings, std::string objectName)
    : mTileset(pTileset)
    , mCelestialObject(std::move(object))
    , mSolarSystem(std::move(pSolarSystem))
    , mSettings(std::move(settings))
    , mObjectName(std::move(objectName)) {

  mEnableLightingConnection = mSettings->mGraphics.pEnableLighting.connect(
      [this](bool /*enabled*/) { mShaderDirty = true; });
  mEnableHDRConnection =
      mSettings->mGraphics.pEnableHDR.connect([this](bool /*enabled*/) { mShaderDirty = true; });

  VistaSceneGraph* pSG = GetVistaSystem()->GetGraphicsManager()->GetSceneGraph();
  mGLNode.reset(pSG->NewOpenGLNode(pSG->GetRoot(), this));
  VistaOpenSGMaterialTools::SetSortKeyOnSubtree(
      mGLNode.get(), static_cast<int>(cs::utils::DrawOrder::eOpaqueItems));

  logger().info("CesiumTilesetRenderer attached to ViSTA scene graph.");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool TilesetRenderer::Do() {
  if (!mCelestialObject) {
    return true;
  }

  if (mShaderDirty) {
    if (mShaderProgram != 0) {
      glDeleteProgram(mShaderProgram);
      mShaderProgram = 0;
    }

    std::string defines = "#version 430\n";

    if (mSettings->mGraphics.pEnableHDR.get()) {
      defines += "#define ENABLE_HDR\n";
    }

    std::string vert = defines + CESIUM_VERT;
    std::string frag = defines + CESIUM_FRAG;

    GLuint vertShader = compileShader(GL_VERTEX_SHADER, vert.c_str());
    GLuint fragShader = compileShader(GL_FRAGMENT_SHADER, frag.c_str());

    if (vertShader && fragShader) {
      mShaderProgram = glCreateProgram();
      glAttachShader(mShaderProgram, vertShader);
      glAttachShader(mShaderProgram, fragShader);
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
        mUniforms.modelMatrix       = glGetUniformLocation(mShaderProgram, "uModelMatrix");
        mUniforms.viewMatrix        = glGetUniformLocation(mShaderProgram, "uViewMatrix");
        mUniforms.projectionMatrix  = glGetUniformLocation(mShaderProgram, "uProjectionMatrix");
        mUniforms.baseColorTexture  = glGetUniformLocation(mShaderProgram, "uBaseColorTexture");
        mUniforms.hasTexture        = glGetUniformLocation(mShaderProgram, "uHasTexture");
        mUniforms.sunIlluminance    = glGetUniformLocation(mShaderProgram, "uSunIlluminance");
        mUniforms.ambientBrightness = glGetUniformLocation(mShaderProgram, "uAmbientBrightness");
        mUniforms.enableLighting    = glGetUniformLocation(mShaderProgram, "uEnableLighting");
        mUniforms.avgLinearImgIntensity =
            glGetUniformLocation(mShaderProgram, "uAvgLinearImgIntensity");
      }
    }

    glDeleteShader(vertShader);
    glDeleteShader(fragShader);

    mShaderDirty = false;
  }

  if (mShaderProgram == 0) {
    return true;
  }

  cs::utils::FrameStats::ScopedTimer timer("Cesium Tileset Rendering");

  glUseProgram(mShaderProgram);

  std::array<GLfloat, 16> glMatV{};
  std::array<GLfloat, 16> glMatP{};
  glGetFloatv(GL_MODELVIEW_MATRIX, glMatV.data());
  glGetFloatv(GL_PROJECTION_MATRIX, glMatP.data());

  glUniformMatrix4fv(mUniforms.viewMatrix, 1, GL_FALSE, glMatV.data());
  glUniformMatrix4fv(mUniforms.projectionMatrix, 1, GL_FALSE, glMatP.data());

  glm::dmat4 observerToBody = mCelestialObject->getObserverRelativeTransform();
  glm::dvec3 bodyPos(observerToBody[3]);

  auto sunIlluminance = static_cast<float>(mSolarSystem->getSunIlluminance(bodyPos));
  glUniform1f(mUniforms.sunIlluminance, sunIlluminance);

  auto      sunDirection = mSolarSystem->getSunDirection(bodyPos);
  glm::vec3 sunDirGL(sunDirection.x, sunDirection.y, sunDirection.z);
  glUniform3f(
      glGetUniformLocation(mShaderProgram, "uSunDirection"), sunDirGL[0], sunDirGL[1], sunDirGL[2]);

  float ambientBrightness = mSettings->mGraphics.pAmbientBrightness.get();
  glUniform1f(mUniforms.ambientBrightness, ambientBrightness);

  glUniform1i(mUniforms.enableLighting, mSettings->mGraphics.pEnableLighting.get() ? 1 : 0);

  cs::core::Settings::Shading const& shading = mSettings->getShadingForBody(mObjectName);
  float                              avgLinearImgIntensity = shading.pAvgLinearImgIntensity.get();
  glUniform1f(mUniforms.avgLinearImgIntensity, avgLinearImgIntensity);

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

    glm::dmat4 tileToObserver = observerToBody * pData->tileTransform;
    glm::mat4  modelMatrix(tileToObserver);

    glUniformMatrix4fv(mUniforms.modelMatrix, 1, GL_FALSE, glm::value_ptr(modelMatrix));

    {
      cs::utils::FrameStats::ScopedTimer drawTimer("Cesium GPU Draw");
      glBindVertexArray(pData->vao);
      glActiveTexture(GL_TEXTURE0);
      glUniform1i(mUniforms.baseColorTexture, 0);

      for (const DrawBatch& batch : pData->batches) {
        GLuint textureId = 0;
        if (batch.textureSlot >= 0 &&
            batch.textureSlot < static_cast<int32_t>(pData->textures.size())) {
          textureId = pData->textures[batch.textureSlot].textureId;
        }

        glBindTexture(GL_TEXTURE_2D, textureId);
        glUniform1i(mUniforms.hasTexture, textureId != 0 ? 1 : 0);
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

////////////////////////////////////////////////////////////////////////////////////////////////////

TilesetRenderer::~TilesetRenderer() {
  mSettings->mGraphics.pEnableLighting.disconnect(mEnableLightingConnection);
  mSettings->mGraphics.pEnableHDR.disconnect(mEnableHDRConnection);

  if (mShaderProgram) {
    glDeleteProgram(mShaderProgram);
  }

  VistaSceneGraph* pSG = GetVistaSystem()->GetGraphicsManager()->GetSceneGraph();
  pSG->GetRoot()->DisconnectChild(mGLNode.get());
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool TilesetRenderer::GetBoundingBox(VistaBoundingBox& /*bb*/) {
  return false;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::cesiumbodies
