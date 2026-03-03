////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "CesiumTilesetRenderer.hpp"
#include "CesiumUtils.hpp"
#include "logger.hpp"

// CosmoScout core headers
#include "../../../src/cs-core/SolarSystem.hpp" // relative path becase to come out plugin and go deep into source code
#include "../../../src/cs-utils/FrameStats.hpp"
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

in vec3 v_Normal;
in vec3 v_Position;
in vec2 v_UV;
in vec4 v_Color;

uniform sampler2D u_BaseColorTexture;
uniform bool  u_HasTexture;
uniform vec3  u_LightDir;
uniform vec3  u_CameraPos;

layout(location = 0) out vec3 oColor;

const float PI = 3.14159265359;

// --- sRGB to Linear conversion (glTF base color textures are sRGB) ---
vec3 sRGBtoLinear(vec3 srgb) {
    return pow(srgb, vec3(2.2));
}

// --- GGX Normal Distribution Function (microfacet roughness) ---
float D_GGX(float NdotH, float roughness) {
    float a  = roughness * roughness;
    float a2 = a * a;
    float denom = NdotH * NdotH * (a2 - 1.0) + 1.0;
    return a2 / (PI * denom * denom);
}

// --- Schlick Fresnel approximation ---
vec3 F_Schlick(float cosTheta, vec3 F0) {
    return F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);
}

void main() {
    // 1. Get the base color
    vec3 baseColor;
    if (u_HasTexture) {
        baseColor = sRGBtoLinear(texture(u_BaseColorTexture, v_UV).rgb);
    } else {
        baseColor = v_Color.rgb;  // Per-vertex color from material baseColorFactor
    }

    // 2. Prepare vectors
    vec3 N = normalize(v_Normal);
    vec3 L = normalize(u_LightDir);
    vec3 viewVec = u_CameraPos - v_Position;
    float viewLen = length(viewVec);
    vec3 V = (viewLen > 0.001) ? (viewVec / viewLen) : vec3(0.0, 0.0, 1.0);

    vec3 H = normalize(L + V);

    // 3. Dot products (clamped to avoid negative lighting)
    float NdotL = max(dot(N, L), 0.0);
    float NdotH = max(dot(N, H), 0.0);
    float VdotH = max(dot(V, H), 0.0);

    // 4. PBR parameters (hardcoded for terrain/buildings)
    float roughness = 0.7;
    vec3  F0 = vec3(0.04);  // Dielectric reflectance (non-metal)

    // 5. Specular: Cook-Torrance microfacet BRDF (simplified)
    float D = D_GGX(NdotH, roughness);
    vec3  F = F_Schlick(VdotH, F0);
    vec3  specular = D * F * 0.25;

    // 6. Diffuse: Lambertian
    vec3 diffuse = baseColor / PI;

    // 7. Combine: light contribution
    vec3 lightColor = vec3(1.0, 0.98, 0.95);
    float ambient = 0.15;

    vec3 color = baseColor * ambient
               + lightColor * NdotL * (diffuse + specular);

    // 8. HDR OUTPUT SCALING
    // CosmoScout uses HDR rendering with physical luminance values.
    // The tone mapper expects values in the tens of thousands.
    // Without this scale, our 0-1 output appears near-black after tone mapping.
    color *= 10000.0;

    oColor = color;
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

  // Upload light direction and camera position (once per frame)
  glm::vec3 lightDir = glm::normalize(glm::vec3(1.0f, 1.0f, 1.0f));
  glUniform3fv(mLocLightDir, 1, glm::value_ptr(lightDir));

  // Camera is at the origin in observer-relative rendering
  glm::vec3 camPos(0.0f, 0.0f, 0.0f);
  glUniform3fv(mLocCameraPos, 1, glm::value_ptr(camPos));

  // 4. Get the Base Model Matrix (Earth relative to Observer)
  glm::dmat4 observerToEarth = earth->getObserverRelativeTransform();


  // 5. Save and set GL state for our draw
  GLboolean cullEnabled  = glIsEnabled(GL_CULL_FACE);
  GLboolean blendEnabled = glIsEnabled(GL_BLEND);
  glDisable(GL_CULL_FACE);
  glDisable(GL_BLEND);
  glEnable(GL_POLYGON_OFFSET_FILL);   // Bias Cesium depth slightly closer to camera
  glPolygonOffset(-1.0f, -1.0f);     // so it draws in front of the Earth surface

  // 6. Get the list of tiles Cesium wants us to render
  const auto& result = mTileset->getDefaultViewGroup().getViewUpdateResult();
  const auto& tiles  = result.tilesToRenderThisFrame;

  uint32_t tilesDrawn = 0;

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
    //    pTile->getTransform() places this tile in ECEF (64-bit)
    //    observerToEarth transforms ECEF → observer-relative (64-bit)
    //    The cast to mat4 is safe because the result contains small relative values.
    glm::dmat4 tileToObserver = observerToEarth * pTile->getTransform();
    glm::mat4  modelMatrix    = glm::mat4(tileToObserver);


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
    glBindVertexArray(pData->vao);
    glDrawElements(GL_TRIANGLES, pData->indexCount, GL_UNSIGNED_INT, nullptr);

    tilesDrawn++;
  }

  // 9. Restore GL state
  if (cullEnabled)
    glEnable(GL_CULL_FACE);
  if (blendEnabled)
    glEnable(GL_BLEND);
  glDisable(GL_POLYGON_OFFSET_FILL);  // Restore default polygon offset state
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
