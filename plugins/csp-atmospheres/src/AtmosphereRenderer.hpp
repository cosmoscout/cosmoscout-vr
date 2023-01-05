////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSP_ATMOSPHERES_ATMOSPHERE_RENDERER_HPP
#define CSP_ATMOSPHERES_ATMOSPHERE_RENDERER_HPP

#include "../../../src/cs-scene/CelestialObject.hpp"
#include "ModelBase.hpp"
#include "Plugin.hpp"

#include <VistaBase/VistaVectorMath.h>
#include <VistaKernel/DisplayManager/VistaViewport.h>
#include <VistaKernel/GraphicsManager/VistaOpenGLDraw.h>
#include <VistaOGLExt/VistaBufferObject.h>
#include <VistaOGLExt/VistaGLSLShader.h>
#include <VistaOGLExt/VistaTexture.h>
#include <VistaOGLExt/VistaVertexArrayObject.h>

#include <glm/glm.hpp>
#include <memory>
#include <unordered_map>

namespace cs::core {
class EclipseShadowReceiver;
}

namespace cs::graphics {
class ShadowMap;
class HDRBuffer;
} // namespace cs::graphics

namespace csp::atmospheres {

/// This class draws a configurable atmosphere. Just put an OpenGLNode into your SceneGraph at the
/// very same position as your planet. Set its scale to the same size as your planet.
class AtmosphereRenderer : public IVistaOpenGLDraw {
 public:
  explicit AtmosphereRenderer(std::shared_ptr<Plugin::Settings> settings,
      std::shared_ptr<cs::core::EclipseShadowReceiver>          eclipseShadowReceiver);

  void configure(Plugin::Settings::Atmosphere const& settings, glm::dvec3 const& radii);

  /// Updates the current sun position and brightness.
  void setSun(glm::vec3 const& direction, float illuminance);

  /// Set the transformation used to draw the atmosphere.
  void setWorldTransform(glm::dmat4 const& transform);

  /// When set, the shader will make lookups in order to generate light shafts.
  void setShadowMap(std::shared_ptr<cs::graphics::ShadowMap> const& pShadowMap);

  /// When set, this buffer will be used as background texture instead of the current backbuffer.
  void setHDRBuffer(std::shared_ptr<cs::graphics::HDRBuffer> const& pHDRBuffer);

  Plugin::Settings::Atmosphere const& getSettings() const;

  bool Do() override;
  bool GetBoundingBox(VistaBoundingBox& bb) override;

 private:
  void updateShader();

  std::shared_ptr<Plugin::Settings> mPluginSettings;
  // std::unique_ptr<VistaTexture>     mCloudTexture;
  // std::string                       mCloudTextureFile;
  // float                             mCloudHeight    = 0.001F;
  // bool                              mUseClouds      = false;

  glm::dvec3                   mRadii          = glm::dvec3(1.0, 1.0, 1.0);
  glm::dmat4                   mWorldTransform = glm::dmat4(1.0);
  Plugin::Settings::Atmosphere mSettings;

  std::shared_ptr<cs::graphics::ShadowMap>         mShadowMap;
  std::shared_ptr<cs::graphics::HDRBuffer>         mHDRBuffer;
  std::shared_ptr<cs::core::EclipseShadowReceiver> mEclipseShadowReceiver;

  VistaGLSLShader        mAtmoShader;
  VistaVertexArrayObject mQuadVAO;
  VistaBufferObject      mQuadVBO;

  struct GBufferData {
    std::unique_ptr<VistaTexture> mDepthBuffer;
    std::unique_ptr<VistaTexture> mColorBuffer;
  };

  std::unordered_map<VistaViewport*, GBufferData> mGBufferData;

  bool      mShaderDirty    = true;
  float     mSunIlluminance = 1.F;
  glm::vec3 mSunDirection   = glm::vec3(1, 0, 0);

  struct {
    uint32_t sunIntensity   = 0;
    uint32_t sunDir         = 0;
    uint32_t depthBuffer    = 0;
    uint32_t colorBuffer    = 0;
    uint32_t cloudTexture   = 0;
    uint32_t cloudAltitude  = 0;
    uint32_t shadowCascades = 0;
    uint32_t sunIlluminance = 0;

    std::array<uint32_t, 5> shadowMaps{};
    std::array<uint32_t, 5> shadowProjectionMatrices{};

    uint32_t inverseModelViewMatrix           = 0;
    uint32_t inverseModelViewProjectionMatrix = 0;
    uint32_t inverseProjectionMatrix          = 0;
    uint32_t modelViewMatrix                  = 0;
    uint32_t modelMatrix                      = 0;
  } mUniforms;

  std::unique_ptr<ModelBase> mModel;
};

} // namespace csp::atmospheres

#endif // CSP_ATMOSPHERES_ATMOSPHERE_RENDERER_HPP
