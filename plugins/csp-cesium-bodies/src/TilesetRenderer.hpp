////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSP_CESIUM_BODIES_TILESET_RENDERER_HPP
#define CSP_CESIUM_BODIES_TILESET_RENDERER_HPP

#include "../../../src/cs-core/Settings.hpp"
#include "../../../src/cs-scene/CelestialSurface.hpp"
#include "../../../src/cs-scene/IntersectableObject.hpp"
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
class TilesetRenderer : public cs::scene::CelestialSurface,
                        public cs::scene::IntersectableObject,
                        public IVistaOpenGLDraw {
 public:
  double getHeight(glm::dvec2 lngLat) const override;

  bool getIntersection(
      glm::dvec3 const& rayPos, glm::dvec3 const& rayDir, glm::dvec3& pos) const override;

  TilesetRenderer(Cesium3DTilesSelection::Tileset*      pTileset,
      std::shared_ptr<const cs::scene::CelestialObject> object,
      std::shared_ptr<cs::core::SolarSystem>            pSolarSystem,
      std::shared_ptr<cs::core::Settings>               settings,
      std::string                                       objectName);

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

  mutable glm::dvec2 mLastQueryLngLat{std::numeric_limits<double>::quiet_NaN(), 0.0};
  mutable double     mCachedHeight        = 0.0;
  mutable bool       mHeightQueryInFlight = false;

  std::unique_ptr<VistaOpenGLNode> mGLNode;

  GLuint mShaderProgram = 0;
  bool   mShaderDirty   = true;

  int mEnableLightingConnection = -1;
  int mEnableHDRConnection      = -1;

  GLint mLocModelMatrix         = -1;
  GLint mLocViewMatrix          = -1;
  GLint mLocProjectionMatrix    = -1;
  GLint mLocBaseColorTexture    = -1;
  GLint mLocHasTexture          = -1;
  GLint mLocSunIlluminance      = -1;
  GLint mLocAmbientBrightness   = -1;
  GLint mLocEnableLighting      = -1;
  GLint mLocAvgLinearImgIntensity = -1;

  static const char* CESIUM_VERT;
  static const char* CESIUM_FRAG;
};

} // namespace csp::cesiumbodies

#endif // CSP_CESIUM_BODIES_TILESET_RENDERER_HPP
