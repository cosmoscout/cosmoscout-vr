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
class HDRBuffer;
} // namespace cs::graphics

namespace csp::atmospheres {

class ModelBase;

/// This class draws a configurable atmosphere. It will be attached to the celestial object
/// identified by the objectName given to the constructor.
class Atmosphere : public IVistaOpenGLDraw {
 public:
  explicit Atmosphere(std::shared_ptr<Plugin::Settings> pluginSettings,
      std::shared_ptr<cs::core::Settings>               allSettings,
      std::shared_ptr<cs::core::SolarSystem>            solarSystem,
      std::shared_ptr<cs::core::GraphicsEngine> graphicsEngine, std::string objectName);

  ~Atmosphere() override;

  /// Reconfigures the atmosphere and the atmospheric model according to the given settings.
  void configure(Plugin::Settings::Atmosphere const& settings);

  /// If the body this is attached to is visible, this will update the transformation of the
  /// atmosphere according to the current observer position. It will also update the
  /// pApproximateSceneBrightness property of the graphics engine in this case.
  void update(double time);

  bool Do() override;
  bool GetBoundingBox(VistaBoundingBox& bb) override;

 private:
  struct Uniforms {
    uint32_t sunDir                    = 0;
    uint32_t sunInfo                   = 0;
    uint32_t time                      = 0;
    uint32_t depthBuffer               = 0;
    uint32_t colorBuffer               = 0;
    uint32_t waterLevel                = 0;
    uint32_t cloudTexture              = 0;
    uint32_t cloudAltitude             = 0;
    uint32_t limbLuminanceTexture      = 0;
    uint32_t inverseModelViewMatrix    = 0;
    uint32_t inverseProjectionMatrix   = 0;
    uint32_t scaleMatrix               = 0;
    uint32_t modelMatrix               = 0;
    uint32_t modelViewProjectionMatrix = 0;
    uint32_t shadowCoordinates         = 0;

    // Only used by the panorama shader.
    uint32_t atmoPanoUniforms = 0;

    // Only used by the skydome shader.
    uint32_t sunElevation = 0;
  };

  enum class ShaderType { eAtmosphere, ePanorama, eSkyDome };

  void createShader(ShaderType type, VistaGLSLShader& shader, Uniforms& uniforms) const;
  void updateShaders();

  void renderSkyDome(std::string const& name) const;

  std::shared_ptr<Plugin::Settings>                mPluginSettings;
  std::shared_ptr<cs::core::Settings>              mAllSettings;
  std::shared_ptr<cs::core::SolarSystem>           mSolarSystem;
  std::shared_ptr<cs::core::GraphicsEngine>        mGraphicsEngine;
  std::string                                      mObjectName;
  std::unique_ptr<VistaOpenGLNode>                 mAtmosphereNode;
  std::shared_ptr<cs::graphics::HDRBuffer>         mHDRBuffer;
  std::shared_ptr<cs::core::EclipseShadowReceiver> mEclipseShadowReceiver;
  std::unique_ptr<VistaTexture>                    mCloudTexture;
  GLuint                                           mLimbLuminanceTexture = 0;

  glm::dvec3                   mRadii                          = glm::dvec3(1.0, 1.0, 1.0);
  glm::dmat4                   mObserverRelativeTransformation = glm::dmat4(1.0);
  double                       mSceneScale                     = 1.0;
  Plugin::Settings::Atmosphere mSettings;

  int mEnableHDRConnection = -1;

  struct GBufferData {
    std::unique_ptr<VistaTexture> mDepthBuffer;
    std::unique_ptr<VistaTexture> mColorBuffer;
  };

  std::unordered_map<VistaViewport*, GBufferData> mGBufferData;

  bool       mShaderDirty    = true;
  double     mSunIlluminance = 1.0;
  double     mSunLuminance   = 1.0;
  glm::dvec3 mSunDirection   = glm::dvec3(1.0, 0.0, 0.0);
  double     mTime           = 0.0;

  VistaGLSLShader mAtmoShader;
  Uniforms        mAtmoUniforms;

  VistaGLSLShader mPanoShader;
  Uniforms        mPanoUniforms;

  std::unique_ptr<ModelBase> mModel;
};

} // namespace csp::atmospheres

#endif // CSP_ATMOSPHERES_ATMOSPHERE_HPP
