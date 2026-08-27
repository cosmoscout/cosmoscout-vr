////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSP_CESIUM_BODIES_TILESET_RENDERER_HPP
#define CSP_CESIUM_BODIES_TILESET_RENDERER_HPP

#include "../../../src/cs-core/Settings.hpp"
#include "../../../src/cs-scene/CelestialObject.hpp"
#include <GL/glew.h>
#include <VistaKernel/GraphicsManager/VistaOpenGLDraw.h>
#include <VistaKernel/GraphicsManager/VistaOpenGLNode.h>

#include <Cesium3DTilesSelection/Tileset.h>

#include <memory>

namespace cs::core {
class SolarSystem;
} // namespace cs::core

namespace csp::cesiumbodies {

/// Renders Cesium 3D Tiles geometry using a basic Lambertian shader. Hooks into the ViSTA scene
/// graph via IVistaOpenGLDraw so the engine calls our Do() method every frame during the render
/// pass.
class TilesetRenderer : public IVistaOpenGLDraw {
 public:
  TilesetRenderer(Cesium3DTilesSelection::Tileset*      pTileset,
      std::shared_ptr<const cs::scene::CelestialObject> object,
      std::shared_ptr<cs::core::SolarSystem>            pSolarSystem,
      std::shared_ptr<cs::core::Settings> settings, std::string objectName);

  ~TilesetRenderer() override;

  TilesetRenderer(TilesetRenderer const& other) = delete;
  TilesetRenderer(TilesetRenderer&& other)      = delete;

  TilesetRenderer& operator=(TilesetRenderer const& other) = delete;
  TilesetRenderer& operator=(TilesetRenderer&& other)      = delete;

  bool Do() override;
  bool GetBoundingBox(VistaBoundingBox& bb) override;

 private:
  Cesium3DTilesSelection::Tileset*                  mTileset;
  std::shared_ptr<const cs::scene::CelestialObject> mCelestialObject;
  std::shared_ptr<cs::core::SolarSystem>            mSolarSystem;
  std::shared_ptr<cs::core::Settings>               mSettings;

  std::string mObjectName;

  std::unique_ptr<VistaOpenGLNode> mGLNode;

  GLuint mShaderProgram = 0;
  bool   mShaderDirty   = true;

  int mEnableLightingConnection = -1;
  int mEnableHDRConnection      = -1;

  struct {
    GLint modelMatrix           = -1;
    GLint viewMatrix            = -1;
    GLint projectionMatrix      = -1;
    GLint baseColorTexture      = -1;
    GLint hasTexture            = -1;
    GLint sunIlluminance        = -1;
    GLint ambientBrightness     = -1;
    GLint enableLighting        = -1;
    GLint avgLinearImgIntensity = -1;
  } mUniforms;

  static const char* CESIUM_VERT;
  static const char* CESIUM_FRAG;
};

} // namespace csp::cesiumbodies

#endif // CSP_CESIUM_BODIES_TILESET_RENDERER_HPP
