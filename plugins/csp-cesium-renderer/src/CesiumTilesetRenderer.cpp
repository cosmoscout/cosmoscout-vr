////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "CesiumTilesetRenderer.hpp"
#include "CesiumUtils.hpp"
#include "logger.hpp"

// CosmoScout core headers
#include "../../../src/cs-core/SolarSystem.hpp"
#include "../../../src/cs-utils/FrameStats.hpp"
#include "../../../src/cs-utils/convert.hpp"
#include "../../../src/cs-utils/utils.hpp"

// ViSTA headers for scene graph hookup
#include <VistaKernel/GraphicsManager/VistaGroupNode.h>
#include <VistaKernel/GraphicsManager/VistaSceneGraph.h> // <> for exgternal library
#include <VistaKernel/VistaSystem.h>
#include <VistaKernelOpenSGExt/VistaOpenSGMaterialTools.h>
#include <VistaMath/VistaBoundingBox.h>

// GLM for matrix math
#include <glm/gtc/matrix_inverse.hpp> // for glm::inverseTranspose
#include <glm/gtc/type_ptr.hpp>       // for matrix math
#include <glm/gtx/transform.hpp>

// Cesium tile access
#include <Cesium3DTilesSelection/Tile.h>
#include <Cesium3DTilesSelection/TileContent.h>
#include <Cesium3DTilesSelection/TilesetViewGroup.h>

namespace csp::cesiumrenderer {

////////////////////////////////////////////////////////////////////////////////////////////////////
// SHADER SOURCE CODE                                                                             //
////////////////////////////////////////////////////////////////////////////////////////////////////
// static const char* is used to store the shader source code in the .cpp file
const char* CesiumTilesetRenderer::CESIUM_VERT = R"( 
#version 430

uniform mat4 u_ModelMatrix;
uniform mat4 u_ViewMatrix;
uniform mat4 u_ProjectionMatrix;
uniform mat3 u_NormalMatrix;

layout(location = 0) in vec3 a_Position;
layout(location = 1) in vec3 a_Normal;
layout(location = 2) in vec2 a_UV;
layout(location = 3) in vec4 a_Color;

out vec3 v_Normal;
out vec3 v_Position;
out vec2 v_UV;
out vec4 v_Color;

void main() {
    vec4 worldPos = u_ModelMatrix * vec4(a_Position, 1.0);
    v_Position    = worldPos.xyz;
    v_Normal      = u_NormalMatrix * a_Normal;
    v_UV          = a_UV;
    v_Color       = a_Color;
    gl_Position   = u_ProjectionMatrix * u_ViewMatrix * worldPos;
}
)";

////////////////////////////////////////////////////////////////////////////////////////////////////

// fragment shader for every pxl,
const char* CesiumTilesetRenderer::CESIUM_FRAG = R"(
#version 430

in vec2 v_UV;
in vec4 v_Color;

uniform sampler2D u_BaseColorTexture;
uniform bool  u_HasTexture;
uniform float u_SunIlluminance;

layout(location = 0) out vec3 oColor;

const float PI = 3.14159265359;

// --- sRGB to Linear conversion (glTF base color textures are sRGB) ---
vec3 sRGBtoLinear(vec3 srgb) {
    return pow(srgb, vec3(2.2));
}

void main() {
    // Google Photorealistic 3D Tiles are PHOTOGRAMMETRY.
    // Textures already contain baked lighting (sunlight, shadows, AO).
    // We do NOT apply PBR lighting — just pass through the base color
    // with HDR scaling for CosmoScout's tone mapper.

    vec3 baseColor;
    if (u_HasTexture) {
        baseColor = sRGBtoLinear(texture(u_BaseColorTexture, v_UV).rgb);
    } else {
        baseColor = v_Color.rgb;
    }

    // HDR scaling for photogrammetry:
    // These are pre-lit photos, NOT raw albedo. The baked lighting already
    // encodes the BRDF response. We scale to match CosmoScout's HDR range.
    // Factor: sunIlluminance * avgAlbedo / PI, where avgAlbedo ~ 0.06
    // for photogrammetry (pre-baked textures appear darker in linear space).
    oColor = baseColor * u_SunIlluminance * 0.06 / PI;
}
)";

////////////////////////////////////////////////////////////////////////////////////////////////////
// HELPER: Compile a single shader stage                                                          //
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
// CONSTRUCTOR                                                                                    //
////////////////////////////////////////////////////////////////////////////////////////////////////

CesiumTilesetRenderer::CesiumTilesetRenderer( // initialize lost
    Cesium3DTilesSelection::Tileset* pTileset, std::shared_ptr<cs::core::SolarSystem> pSolarSystem)
    : mTileset(pTileset)
    , mSolarSystem(std::move(pSolarSystem)) { // takes psolar and puts it into msolar

  // ---- 1. COMPILE AND LINK THE SHADER PROGRAM ----
  GLuint vert = compileShader(GL_VERTEX_SHADER, CESIUM_VERT);
  GLuint frag = compileShader(GL_FRAGMENT_SHADER, CESIUM_FRAG);

  if (vert && frag) {
    mShaderProgram =
        glCreateProgram(); // creates new empty container and gives back new id and stores it
    glAttachShader(mShaderProgram, vert); // attaches the vertex shader to the program
    glAttachShader(mShaderProgram, frag); // attaches the fragment shader to the program
    glLinkProgram(mShaderProgram);        // links the program together

    GLint linked = 0;
    glGetProgramiv(mShaderProgram, GL_LINK_STATUS, &linked); // manage link failure
    if (!linked) {
      char infoLog[512];
      glGetProgramInfoLog(mShaderProgram, 512, nullptr, infoLog);
      logger().error("Shader link failed: {}", infoLog);
      glDeleteProgram(mShaderProgram);
      mShaderProgram = 0;
    } else {
      // Cache uniform locations — only done once, not per-frame
      mLocModelMatrix      = glGetUniformLocation(mShaderProgram, "u_ModelMatrix");
      mLocViewMatrix       = glGetUniformLocation(mShaderProgram, "u_ViewMatrix");
      mLocProjectionMatrix = glGetUniformLocation(mShaderProgram, "u_ProjectionMatrix");
      mLocNormalMatrix     = glGetUniformLocation(mShaderProgram, "u_NormalMatrix");
      mLocBaseColorTexture = glGetUniformLocation(mShaderProgram, "u_BaseColorTexture");
      mLocHasTexture       = glGetUniformLocation(mShaderProgram, "u_HasTexture");
      mLocLightDir         = glGetUniformLocation(mShaderProgram, "u_LightDir");
      mLocCameraPos        = glGetUniformLocation(mShaderProgram, "u_CameraPos");
      mLocSunIlluminance   = glGetUniformLocation(mShaderProgram, "u_SunIlluminance");

      logger().info("Cesium shader compiled and linked successfully.");
      logger().info("  Uniform locations: Model={}, View={}, Proj={}, Normal={}, Tex={}, "
                    "HasTex={}, Light={}, Cam={}",
          mLocModelMatrix, mLocViewMatrix, mLocProjectionMatrix, mLocNormalMatrix,
          mLocBaseColorTexture, mLocHasTexture, mLocLightDir, mLocCameraPos);
    }
  }

  // Delete shader objects — they're baked into the program now.
  glDeleteShader(vert);
  glDeleteShader(frag);

  // ---- 2. HOOK INTO THE VISTA SCENE GRAPH ----
  VistaSceneGraph* pSG = GetVistaSystem()->GetGraphicsManager()->GetSceneGraph();
  mGLNode.reset(pSG->NewOpenGLNode(pSG->GetRoot(), this));
  VistaOpenSGMaterialTools::SetSortKeyOnSubtree(
      mGLNode.get(), static_cast<int>(cs::utils::DrawOrder::eOpaqueItems));

  logger().info("CesiumTilesetRenderer attached to ViSTA scene graph.");
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// THE DRAW LOOP                                                                                  //
////////////////////////////////////////////////////////////////////////////////////////////////////

bool CesiumTilesetRenderer::Do() {
  // 1. Guard: If the shader failed to compile, don't draw.
  auto earth = mSolarSystem->getObject("Earth");
  if (mShaderProgram == 0 || !earth) {
    return true;
  }

  cs::utils::FrameStats::ScopedTimer timer("Cesium Tileset Rendering");

  static int frameCounter = 0;
  frameCounter++;

  // 2. Activate our shader program
  glUseProgram(mShaderProgram);

  // 3. Get ViSTA's current View and Projection matrices from the OpenGL state.
  std::array<GLfloat, 16> glMatV{};
  std::array<GLfloat, 16> glMatP{};
  glGetFloatv(GL_MODELVIEW_MATRIX, glMatV.data());
  glGetFloatv(GL_PROJECTION_MATRIX, glMatP.data());

  // Upload view/projection once per frame
  glUniformMatrix4fv(mLocViewMatrix, 1, GL_FALSE, glMatV.data());
  glUniformMatrix4fv(mLocProjectionMatrix, 1, GL_FALSE, glMatP.data());

  // Upload light direction from CosmoScout SolarSystem (matches csp-simple-bodies pattern)
  glm::dmat4 observerToEarth = earth->getObserverRelativeTransform();
  glm::dvec3 earthPos        = glm::dvec3(observerToEarth[3]);

  // Upload physically-correct sun illuminance (CosmoScout's HDR system)
  float sunIlluminance = static_cast<float>(mSolarSystem->getSunIlluminance(earthPos));
  glUniform1f(mLocSunIlluminance, sunIlluminance);

  // Camera is at the origin in observer-relative rendering
  // (u_LightDir and u_CameraPos are no longer used — photogrammetry shader is unlit)

  // 5. Save and set GL state for our draw
  //    CosmoScout uses reverse-Z depth (GL_GEQUAL) and flipped winding (GL_CW).
  //    The projection flips triangle winding, so SetupGLNode sets GL_CW = front.
  //    Rather than fighting the winding convention, just disable face culling.
  GLint     prevDepthFunc;
  GLboolean cullEnabled  = glIsEnabled(GL_CULL_FACE);
  GLboolean blendEnabled = glIsEnabled(GL_BLEND);

  glGetIntegerv(GL_DEPTH_FUNC, &prevDepthFunc);

  glDepthFunc(GL_GEQUAL);  // CosmoScout's reverse-Z: near=1.0, far=0.0
  glDisable(GL_CULL_FACE); // Disable culling — avoids winding order conflicts
  glDisable(GL_BLEND);

  // 6. Get the list of tiles Cesium wants us to render
  const auto& result = mTileset->getDefaultViewGroup().getViewUpdateResult();
  const auto& tiles  = result.tilesToRenderThisFrame;

  uint32_t tilesDrawn = 0;

  // Periodic diagnostic logging (every 500 frames)
  static int diagCounter = 0;
  if (diagCounter % 500 == 0 && !tiles.empty()) {
    auto&  observer = mSolarSystem->getObserver();
    double camDist  = glm::length(observer.getPosition());
    double scale    = observer.getScale();
    logger().warn("[DIAG] SunIll={:.0f}, CamDist={:.0f}m, Scale={:.3e}, TilesSelected={}",
        sunIlluminance, camDist, scale, tiles.size());
  }
  diagCounter++;

  for (auto const& pTilePointer : tiles) {
    const auto* pTile = pTilePointer.get();

    // Skip tiles that haven't finished loading
    auto state = pTile->getState();
    if (state != Cesium3DTilesSelection::TileLoadState::ContentLoaded &&
        state != Cesium3DTilesSelection::TileLoadState::Done) {
      continue;
    }

    // Get the render content (our CesiumRenderData pointer)
    auto* pRenderContent = pTile->getContent().getRenderContent();
    if (!pRenderContent)
      continue;

    auto* pData = static_cast<CesiumRenderData*>(pRenderContent->getRenderResources());
    if (!pData || pData->vao == 0)
      continue;

    // 7. Compute per-tile model matrix (observer-relative):
    //    pData->tileTransform is the CORRECTED tile-to-ECEF transform
    //    (with applyRtcCenter + applyGltfUpAxisTransform already applied).
    //    observerToEarth transforms ECEF → observer-relative (64-bit).
    //    The cast to mat4 is safe because the result contains small relative values.
    glm::dmat4 tileToObserver = observerToEarth * pData->tileTransform;
    glm::mat4  modelMatrix    = glm::mat4(tileToObserver);

    // Periodic tile diagnostic (first valid tile, every 500 frames)
    if ((diagCounter - 1) % 500 == 0 && tilesDrawn == 0) {
      logger().warn(
          "[DIAG] Tile0 mat diag=({:.6f},{:.6f},{:.6f},{:.6f}), col3=({:.4f},{:.4f},{:.4f})",
          modelMatrix[0][0], modelMatrix[1][1], modelMatrix[2][2], modelMatrix[3][3],
          modelMatrix[3][0], modelMatrix[3][1], modelMatrix[3][2]);
    }

    glUniformMatrix4fv(mLocModelMatrix, 1, GL_FALSE, glm::value_ptr(modelMatrix));

    // 7b. Compute and upload the normal matrix (inverse-transpose of model matrix)
    glm::dmat3 normalMatrixD = glm::dmat3(glm::inverseTranspose(tileToObserver));
    glm::mat3  normalMatrix  = glm::mat3(normalMatrixD);

    glUniformMatrix3fv(mLocNormalMatrix, 1, GL_FALSE, glm::value_ptr(normalMatrix));

    // 7c. Bind texture (if this tile has one)
    if (pData->textureId != 0) {
      glActiveTexture(GL_TEXTURE0);
      glBindTexture(GL_TEXTURE_2D, pData->textureId);
      glUniform1i(mLocBaseColorTexture, 0); // Use texture unit 0
      glUniform1i(mLocHasTexture, 1);       // true
    } else {
      glUniform1i(mLocHasTexture, 0); // false
    }

    // 8. Bind the tile's VAO and draw!
    {
      cs::utils::FrameStats::ScopedTimer drawTimer("Cesium GPU Draw");
      glBindVertexArray(pData->vao);
      glDrawElements(GL_TRIANGLES, pData->indexCount, GL_UNSIGNED_INT, nullptr);
    }

    tilesDrawn++;
  }

  // 9. Restore GL state PERFECTLY — failing to do so corrupts the tone mapper
  glDepthFunc(prevDepthFunc); // CRITICAL: restore reverse-Z depth function
  if (cullEnabled)
    glEnable(GL_CULL_FACE); // Restore face culling if it was on
  if (blendEnabled)
    glEnable(GL_BLEND);
  glBindVertexArray(0);
  glUseProgram(0);

  // 10. Throttled diagnostic logging (every 100 frames)
  if (frameCounter % 100 == 0) {
    if (tilesDrawn > 0) {
      logger().info("Drawing {} Cesium tiles.", tilesDrawn);
    } else if (!tiles.empty()) {
      logger().warn(
          "Cesium has {} tiles selected, but 0 reached draw! Check tile states.", tiles.size());
    }
  }

  return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// HELPER: Möller–Trumbore ray-triangle intersection                                              //
////////////////////////////////////////////////////////////////////////////////////////////////////

/// Tests whether a ray hits a triangle. Returns true if hit, and sets 'tOut' to the parametric
/// distance along the ray (hit point = origin + tOut * direction).
static bool rayTriangleIntersect(glm::dvec3 const& origin, glm::dvec3 const& dir,
    glm::dvec3 const& v0, glm::dvec3 const& v1, glm::dvec3 const& v2, double& tOut) {

  const double EPSILON = 1e-9;

  glm::dvec3 e1 = v1 - v0;
  glm::dvec3 e2 = v2 - v0;
  glm::dvec3 h  = glm::cross(dir, e2);
  double     a  = glm::dot(e1, h);

  // Ray is parallel to the triangle
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
  double     v = f * glm::dot(dir, q);

  if (v < 0.0 || u + v > 1.0) {
    return false;
  }

  double t = f * glm::dot(e2, q);

  if (t > EPSILON) {
    tOut = t;
    return true;
  }

  return false;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// CelestialSurface: getHeight                                                                    //
////////////////////////////////////////////////////////////////////////////////////////////////////

double CesiumTilesetRenderer::getHeight(glm::dvec2 lngLat) const {
  // DLL VERSION MARKER — proves at runtime that this rebuilt DLL is loaded.
  // If you see this in the log, the getHeight()=0 bypass IS active.
  static int sCallCount = 0;
  if (++sCallCount % 300 == 1) {
    logger().warn("[CESIUM_GETHEIGHT_V2] getHeight() BYPASS active — returning 0.0 "
                  "(call #{})",
        sCallCount);
  }

  // Returning 0 disables surface collision. CosmoScout navigates relative to the
  // WGS84 ellipsoid, allowing the camera to freely approach the surface.
  return 0.0;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// IntersectableObject: getIntersection                                                           //
////////////////////////////////////////////////////////////////////////////////////////////////////

bool CesiumTilesetRenderer::getIntersection(
    glm::dvec3 const& rayPos, glm::dvec3 const& rayDir, glm::dvec3& pos) const {
  if (!mTileset) {
    return false;
  }

  auto earth = mSolarSystem->getObject("Earth");
  if (!earth) {
    return false;
  }

  // The ray comes in relative to the CelestialObject's coordinate system (ECEF)
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

    // Transform ray into tile-local space
    // MUST use pData->tileTransform (corrected with RTC center + up-axis),
    // because vertex positions were baked in this coordinate space.
    glm::dmat4 tileXform    = pData->tileTransform;
    glm::dmat4 invTileXform = glm::inverse(tileXform);

    glm::dvec3 localOrigin = glm::dvec3(invTileXform * glm::dvec4(rayPos, 1.0));
    glm::dvec3 localDir    = glm::normalize(glm::dvec3(invTileXform * glm::dvec4(rayDir, 0.0)));

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

      double t = 0.0;
      if (rayTriangleIntersect(localOrigin, localDir, v0, v1, v2, t)) {
        if (t > 0.0 && t < closestT) {
          closestT = t;

          // Convert hit point back to ECEF
          glm::dvec3 localHit = localOrigin + t * localDir;
          pos                 = glm::dvec3(tileXform * glm::dvec4(localHit, 1.0));
          foundHit            = true;
        }
      }
    }
  }

  return foundHit;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// DESTRUCTOR                                                                                     //
////////////////////////////////////////////////////////////////////////////////////////////////////

CesiumTilesetRenderer::~CesiumTilesetRenderer() {
  if (mShaderProgram) {
    glDeleteProgram(mShaderProgram);
  }

  // Clean up scene graph connection
  VistaSceneGraph* pSG = GetVistaSystem()->GetGraphicsManager()->GetSceneGraph();
  pSG->GetRoot()->DisconnectChild(mGLNode.get());
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool CesiumTilesetRenderer::GetBoundingBox(VistaBoundingBox& /*bb*/) {
  return false;
}

} // namespace csp::cesiumrenderer
