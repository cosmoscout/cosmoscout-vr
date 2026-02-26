////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSP_CESIUM_RENDERER_CESIUM_TILESET_RENDERER_HPP
#define CSP_CESIUM_RENDERER_CESIUM_TILESET_RENDERER_HPP

#include <GL/glew.h> // talks to graphics card
#include <VistaKernel/GraphicsManager/VistaOpenGLDraw.h> // talks to vista
#include <VistaKernel/GraphicsManager/VistaOpenGLNode.h> // talks to vistas scengraph nodes

#include <Cesium3DTilesSelection/Tileset.h> // for cesium tileset

#include <memory> // for smart pointers

namespace cs::core {
class SolarSystem; // forward delecaration, , for finding earth and camera
} // namespace cs::core

namespace csp::cesiumrenderer {

/// Renders Cesium 3D Tiles geometry using a basic Lambertian shader.
/// Hooks into the ViSTA scene graph via IVistaOpenGLDraw so the engine
/// calls our Do() method every frame during the render pass.
class CesiumTilesetRenderer : public IVistaOpenGLDraw { // cesiumtileset renderer is a vistaopendraw object
 public:
  CesiumTilesetRenderer(Cesium3DTilesSelection::Tileset* pTileset,  // to get 3d data from cesium
      std::shared_ptr<cs::core::SolarSystem>             pSolarSystem); //shared pointer to figure out loctn

  ~CesiumTilesetRenderer() override; // destructor to clean up

  CesiumTilesetRenderer(CesiumTilesetRenderer const& other) = delete;  // to prevent copying and deleting and throws error
  CesiumTilesetRenderer(CesiumTilesetRenderer&& other)      = delete;

  CesiumTilesetRenderer& operator=(CesiumTilesetRenderer const& other) = delete;
  CesiumTilesetRenderer& operator=(CesiumTilesetRenderer&& other)      = delete;

  /// Called by ViSTA every frame during the render pass.
  bool Do() override; // vista call this 60 fps to draw
  bool GetBoundingBox(VistaBoundingBox& bb) override; //frustum culling, to return box so vista knows

 private:
  Cesium3DTilesSelection::Tileset*      mTileset;  // saving items so to use later in do loop
  std::shared_ptr<cs::core::SolarSystem> mSolarSystem;

  std::unique_ptr<VistaOpenGLNode> mGLNode; // node carruying the vista stage, and removes from memory after use

  GLuint mShaderProgram = 0; //storing shader id 

  // Cached uniform locations
  GLint mLocModelMatrix      = -1; // where is earth 
  GLint mLocViewMatrix       = -1; // where is camera
  GLint mLocProjectionMatrix = -1; // feld of view

  // Shader source code (defined as static constants in the .cpp)
  static const char* CESIUM_VERT;  // shader language glsl , calculates math for 3d points
  static const char* CESIUM_FRAG; // for colors and textures
};

} // namespace csp::cesiumrenderer

#endif // CSP_CESIUM_RENDERER_CESIUM_TILESET_RENDERER_HPP
