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
#include <VistaKernel/GraphicsManager/VistaSceneGraph.h> // <> for exgternal library 
#include <VistaKernel/GraphicsManager/VistaGroupNode.h>
#include <VistaKernel/VistaSystem.h>
#include <VistaKernelOpenSGExt/VistaOpenSGMaterialTools.h>
#include <VistaMath/VistaBoundingBox.h>

// GLM for matrix math
#include <glm/gtc/type_ptr.hpp> // for matrix math
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

layout(location = 0) in vec3 a_Position;
layout(location = 1) in vec3 a_Normal;

out vec3 v_Normal;
out vec3 v_Position;

void main() {
    vec4 worldPos = u_ModelMatrix * vec4(a_Position, 1.0);
    v_Position    = worldPos.xyz;
    v_Normal      = mat3(u_ModelMatrix) * a_Normal;
    gl_Position   = u_ProjectionMatrix * u_ViewMatrix * worldPos;
}
)";

////////////////////////////////////////////////////////////////////////////////////////////////////

//fragment shader for every pxl, 
const char* CesiumTilesetRenderer::CESIUM_FRAG = R"(
#version 430

in vec3 v_Normal;
in vec3 v_Position;

layout(location = 0) out vec3 oColor;

void main() {
    // Basic Lambertian: use normal to compute a simple light direction
    vec3 lightDir = normalize(vec3(1.0, 1.0, 1.0));
    vec3 N        = normalize(v_Normal);
    float diffuse = max(dot(N, lightDir), 0.0);

    // Ambient baseline so shadowed areas aren't pitch black
    float ambient = 0.15;

    // THE OPTIMIZATION TRAP FIX:
    // GPU compilers are aggressive optimizers. If u_ModelMatrix is not
    // "visibly" used in the fragment shader, the compiler may decide
    // that u_ModelMatrix is dead code and DELETE it from the program.
    // This would cause glGetUniformLocation("u_ModelMatrix") to return -1
    // on the CPU side, silently breaking our entire rendering.
    // By adding a mathematically insignificant amount of v_Position
    // (which depends on u_ModelMatrix), we force the compiler to keep it.
    ambient += v_Position.x * 0.000000001;

    // NEON YELLOW — visible from 8,000 km orbit
    oColor = vec3(1.0, 1.0, 0.0) * (diffuse + ambient);
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
    Cesium3DTilesSelection::Tileset* pTileset,
    std::shared_ptr<cs::core::SolarSystem> pSolarSystem)
    : mTileset(pTileset)
    , mSolarSystem(std::move(pSolarSystem)) { // takes psolar and puts it into msolar

  // ---- 1. COMPILE AND LINK THE SHADER PROGRAM ----
  GLuint vert = compileShader(GL_VERTEX_SHADER, CESIUM_VERT); 
  GLuint frag = compileShader(GL_FRAGMENT_SHADER, CESIUM_FRAG);

  if (vert && frag) {
    mShaderProgram = glCreateProgram(); //creates new empty container and gives back new id and stores it 
    glAttachShader(mShaderProgram, vert); // attaches the vertex shader to the program
    glAttachShader(mShaderProgram, frag); // attaches the fragment shader to the program
    glLinkProgram(mShaderProgram); // links the program together

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
      logger().info("Cesium shader compiled and linked successfully.");
      logger().info("  u_ModelMatrix loc={}, u_ViewMatrix loc={}, u_ProjectionMatrix loc={}",
          mLocModelMatrix, mLocViewMatrix, mLocProjectionMatrix);
    }
  }

  // Delete shader objects — they're baked into the program now.
  glDeleteShader(vert);
  glDeleteShader(frag);

  // ---- 2. HOOK INTO THE VISTA SCENE GRAPH ----
  VistaSceneGraph* pSG = GetVistaSystem()->GetGraphicsManager()->GetSceneGraph();
  mGLNode.reset(pSG->NewOpenGLNode(pSG->GetRoot(), this));
  VistaOpenSGMaterialTools::SetSortKeyOnSubtree(
      mGLNode.get(), static_cast<int>(cs::utils::DrawOrder::ePlanets));

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

  // 4. Get the Base Model Matrix (Earth relative to Observer)
  glm::dmat4 observerToEarth = earth->getObserverRelativeTransform();

  // THE MEGA-SCALE HACK: 10,000x scale so tiny glTF tiles are visible from orbit
  glm::dmat4 megaScale = glm::scale(glm::dmat4(1.0), glm::dvec3(10000.0));

  // 5. Save and set GL state for our draw
  GLboolean cullEnabled  = glIsEnabled(GL_CULL_FACE);
  GLboolean blendEnabled = glIsEnabled(GL_BLEND);
  glDisable(GL_CULL_FACE);
  glDisable(GL_BLEND);

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
    if (!pRenderContent) continue;

    auto* pData = static_cast<CesiumRenderData*>(pRenderContent->getRenderResources());
    if (!pData || pData->vao == 0) continue;

    // 7. Compute per-tile model matrix:
    //    observerToEarth places Earth relative to camera
    //    pTile->getTransform() places this tile relative to Earth (ECEF)
    //    megaScale inflates the geometry so we can see it from 8,000 km
    glm::dmat4 tileToObserver = observerToEarth * pTile->getTransform() * megaScale;
    glm::mat4  modelMatrix    = glm::mat4(tileToObserver);

    glUniformMatrix4fv(mLocModelMatrix, 1, GL_FALSE, glm::value_ptr(modelMatrix));

    // 8. Bind the tile's VAO and draw!
    glBindVertexArray(pData->vao);
    glDrawElements(GL_TRIANGLES, pData->indexCount, GL_UNSIGNED_INT, nullptr);

    tilesDrawn++;
  }

  // 9. Restore GL state
  if (cullEnabled)  glEnable(GL_CULL_FACE);
  if (blendEnabled) glEnable(GL_BLEND);
  glBindVertexArray(0);
  glUseProgram(0);

  // 10. Throttled diagnostic logging (every 100 frames)
  static int frameCounter = 0;
  if (frameCounter++ % 100 == 0) {
    if (tilesDrawn > 0) {
      logger().info("Drawing {} Cesium tiles.", tilesDrawn);
    } else if (!tiles.empty()) {
      logger().warn("Cesium has {} tiles selected, but 0 reached draw! Check tile states.",
          tiles.size());
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





