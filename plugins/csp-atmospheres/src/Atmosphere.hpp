////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSP_ATMOSPHERES_ATMOSPHERE_HPP
#define CSP_ATMOSPHERES_ATMOSPHERE_HPP

#include "Plugin.hpp"

#include <VistaKernel/GraphicsManager/VistaOpenGLDraw.h>
#include <VistaOGLExt/VistaGLSLShader.h>

namespace cs::core {
class SolarSystem;
class GraphicsEngine;
class EclipseShadowReceiver;
} // namespace cs::core

namespace cs::graphics {
class ShadowMap;
class HDRBuffer;
} // namespace cs::graphics

namespace csp::atmospheres {

class ModelBase;

/// This class draws a configurable atmosphere. Just put an OpenGLNode into your SceneGraph at the
/// very same position as your planet. Set its scale to the same size as your planet.
class Atmosphere : public IVistaOpenGLDraw {
 public:
  explicit Atmosphere(std::shared_ptr<Plugin::Settings> pluginSettings,
      std::shared_ptr<cs::core::Settings>               allSettings,
      std::shared_ptr<cs::core::SolarSystem>            solarSystem,
      std::shared_ptr<cs::core::GraphicsEngine> graphicsEngine, std::string objectName);

  ~Atmosphere();

  void configure(Plugin::Settings::Atmosphere const& settings);

  void update();

  bool Do() override;
  bool GetBoundingBox(VistaBoundingBox& bb) override;

 private:
  void updateShader();

  std::shared_ptr<Plugin::Settings>         mPluginSettings;
  std::shared_ptr<cs::core::Settings>       mAllSettings;
  std::shared_ptr<cs::core::SolarSystem>    mSolarSystem;
  std::shared_ptr<cs::core::GraphicsEngine> mGraphicsEngine;
  std::string                               mObjectName;
  std::unique_ptr<VistaOpenGLNode>          mAtmosphereNode;
  // std::shared_ptr<cs::graphics::ShadowMap> mShadowMap;
  std::shared_ptr<cs::graphics::HDRBuffer> mHDRBuffer;
  // std::shared_ptr<cs::core::EclipseShadowReceiver> mEclipseShadowReceiver;

  // std::unique_ptr<VistaTexture>     mCloudTexture;
  // std::string                       mCloudTextureFile;
  // float                             mCloudHeight    = 0.001F;
  // bool                              mUseClouds      = false;

  glm::dvec3                   mRadii          = glm::dvec3(1.0, 1.0, 1.0);
  glm::dmat4                   mWorldTransform = glm::dmat4(1.0);
  Plugin::Settings::Atmosphere mSettings;

  int mEnableShadowsConnection = -1;
  int mEnableHDRConnection     = -1;

  VistaGLSLShader mAtmoShader;

  struct GBufferData {
    std::unique_ptr<VistaTexture> mDepthBuffer;
    std::unique_ptr<VistaTexture> mColorBuffer;
  };

  std::unordered_map<VistaViewport*, GBufferData> mGBufferData;

  bool      mShaderDirty    = true;
  float     mSunIlluminance = 1.F;
  glm::vec3 mSunDirection   = glm::vec3(1, 0, 0);

  struct {
    uint32_t sunDir         = 0;
    uint32_t sunIlluminance = 0;
    uint32_t depthBuffer    = 0;
    uint32_t colorBuffer    = 0;
    uint32_t waterLevel     = 0;
    // uint32_t cloudTexture   = 0;
    // uint32_t cloudAltitude  = 0;
    // uint32_t shadowCascades = 0;

    // std::array<uint32_t, 5> shadowMaps{};
    // std::array<uint32_t, 5> shadowProjectionMatrices{};

    uint32_t inverseModelViewMatrix           = 0;
    uint32_t inverseModelViewProjectionMatrix = 0;
    uint32_t inverseProjectionMatrix          = 0;
    // uint32_t modelViewMatrix                  = 0;
    // uint32_t modelMatrix                      = 0;
  } mUniforms;

  std::unique_ptr<ModelBase> mModel;
};

} // namespace csp::atmospheres

#endif // CSP_ATMOSPHERES_ATMOSPHERE_HPP
