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

#include "utils.hpp"
#include "Tree.hpp"

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
  enum class ShaderType { eAtmosphere, eSkyDome };

  void createShader(ShaderType type, VistaGLSLShader& shader, utils::Uniforms& uniforms) const;
  void updateShaders();

  void renderSkyDome(std::string const& name) const;

  void BuildOctree();

  std::shared_ptr<Plugin::Settings>                mPluginSettings;
  std::shared_ptr<cs::core::Settings>              mAllSettings;
  std::shared_ptr<cs::core::SolarSystem>           mSolarSystem;
  std::shared_ptr<cs::core::GraphicsEngine>        mGraphicsEngine;
  std::string                                      mObjectName;
  std::unique_ptr<VistaOpenGLNode>                 mAtmosphereNode;
  std::shared_ptr<cs::graphics::HDRBuffer>         mHDRBuffer;
  std::shared_ptr<cs::core::EclipseShadowReceiver> mEclipseShadowReceiver;

  std::vector<float>                               mNoiseData;
  std::vector<float>                               mNoiseData2D;

  std::unique_ptr<VistaTexture>                    mCloudTexture;     // earth-clouds.jpg
  std::unique_ptr<VistaTexture>                    mCloudTypeTexture; // cloudTop.png
  GLuint                                           mNoiseTexture = 0;
  GLuint                                           mNoiseTexture2D = 0;
  GLuint                                           mLimbLuminanceTexture = 0;
  GLuint                                           mCloudTreeBuffer = 0;

  const int resx = 32, resy = 32, resz = 32, channels = 3;
  const int resz2 = 256, resy2 = 256;
  glm::dvec3                   mRadii                          = glm::dvec3(1.0, 1.0, 1.0);
  glm::dmat4                   mObserverRelativeTransformation = glm::dmat4(1.0);
  double                       mSceneScale                     = 1.0;
  Plugin::Settings::Atmosphere mSettings;

  
  double                        mPlanetRadius;

  int mEnableHDRConnection = -1;

  bool       mShaderDirty    = true;
  double     mSunIlluminance = 1.0;
  double     mSunLuminance   = 1.0;
  glm::dvec3 mSunDirection   = glm::dvec3(1.0, 0.0, 0.0);
  double     mTime           = 0.0;

  VistaGLSLShader mAtmoShader;
  utils::Uniforms        mAtmoUniforms;

  std::unique_ptr<ModelBase> mModel;

  // Octree for 3D cloud raymarcher
  std::unique_ptr<Tree> mCloudTree;
};

} // namespace csp::atmospheres

#endif // CSP_ATMOSPHERES_ATMOSPHERE_HPP
