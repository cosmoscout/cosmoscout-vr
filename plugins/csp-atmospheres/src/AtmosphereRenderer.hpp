////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CSP_ATMOSPHERE_RENDERER_HPP
#define CSP_ATMOSPHERE_RENDERER_HPP

#include "../../../src/cs-scene/CelestialObject.hpp"
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

namespace cs::graphics {
class ShadowMap;
class HDRBuffer;
} // namespace cs::graphics

namespace csp::atmospheres {

/// This class draws a configurable atmosphere. Just put an OpenGLNode into your SceneGraph at the
/// very same position as your planet. Set its scale to the same size as your planet.
class AtmosphereRenderer : public IVistaOpenGLDraw {
 public:
  explicit AtmosphereRenderer(std::shared_ptr<Plugin::Settings> settings);

  /// Updates the current sun position and brightness.
  void setSun(glm::vec3 const& direction, float illuminance);

  /// Set the planet's radii.
  void setRadii(glm::dvec3 const& radii);

  /// Set the transformation used to draw the atmosphere.
  void setWorldTransform(glm::dmat4 const& transform);

  /// When set, the shader will draw this texture at the given altitude.
  void setClouds(std::string const& textureFile, float height);

  /// When set, the shader will make lookups in order to generate light shafts.
  void setShadowMap(std::shared_ptr<cs::graphics::ShadowMap> const& pShadowMap);

  /// When set, this buffer will be used as background texture instead of the current backbuffer.
  void setHDRBuffer(std::shared_ptr<cs::graphics::HDRBuffer> const& pHDRBuffer);

  /// Returns a value [0..1] which approximates the overall brightness of the atmosphere. Will be
  /// close to zero in outer space or in the planets shadow, close to one on the bright surface of
  /// the planet. It uses the camera data from the last rendering call. Use this value for fake HDR
  /// effects, such as reducing star brightness.
  float getApproximateSceneBrightness() const;

  /// How many samples are taken along the view ray. Trades quality for performance. Default is 15.
  int  getPrimaryRaySteps() const;
  void setPrimaryRaySteps(int iValue);

  /// How many samples are taken along the sun rays. Trades quality for performance. Default is 3.
  int  getSecondaryRaySteps() const;
  void setSecondaryRaySteps(int iValue);

  /// The maximum height of the atmosphere above the planets surface relative to the planets radius.
  /// Default depends on the preset; for Earth 60.0 / 6360.0 is assumed.
  float getAtmosphereHeight() const;
  void  setAtmosphereHeight(float dValue);

  /// The scale height for Mie scattering above the planets surface relative to the planets radius.
  /// Default depends on the preset; for Earth 1.2 / 6360.0 is assumed.
  float getMieHeight() const;
  void  setMieHeight(float dValue);

  /// The Mie scattering values. Default depends on the preset; for Earth (21.0, 21.0, 21.0) is
  /// assumed.
  glm::vec3 getMieScattering() const;
  void      setMieScattering(const glm::vec3& vValue);

  /// The Mie scattering anisotropy. Default depends on the preset; for Earth 0.76 is assumed.
  float getMieAnisotropy() const;
  void  setMieAnisotropy(float dValue);

  /// The scale height for Rayleigh scattering above the planets surface relative to the planets
  /// radius. Default depends on the preset; for Earth 8.0 / 6360.0 is assumed.
  float getRayleighHeight() const;
  void  setRayleighHeight(float dValue);

  /// The Rayleigh scattering values. Default depends on the preset; for Earth (5.8, 13.5, 21.1) is
  /// assumed.
  glm::vec3 getRayleighScattering() const;
  void      setRayleighScattering(const glm::vec3& vValue);

  /// The Rayleigh scattering anisotropy. Default depends on the preset; for Earth 0.0 is assumed.
  float getRayleighAnisotropy() const;
  void  setRayleighAnisotropy(float dValue);

  /// If true, an artificial disc is drawn in the suns direction.
  bool getDrawSun() const;
  void setDrawSun(bool bEnable);

  /// If true, a ocean is drawn at water level.
  bool getDrawWater() const;
  void setDrawWater(bool bEnable);

  /// The height of the ocean level. In atmosphere space, that means 0 equals sea level, larger
  /// values increase the sea level. If set to AtmosphereHeight, the ocean will be at atmosphere the
  /// boundary.
  float getWaterLevel() const;
  void  setWaterLevel(float fValue);

  /// A value of 0 will multiply the planet surface with the extinction factor of the sun color,
  /// making the night side completely dark. A value of 1 will result in no extinction of the
  /// incoming light.
  float getAmbientBrightness() const;
  void  setAmbientBrightness(float fValue);

  /// If true, tonemapping is performed on the atmospheric color.
  bool getUseToneMapping() const;
  void setUseToneMapping(bool bEnable, float fExposure, float fGamma);

  /// If true, the depth buffer is assumed to contain linear depth values. This significantly
  /// reduces artifacts for large scenes.
  bool getUseLinearDepthBuffer() const;
  void setUseLinearDepthBuffer(bool bEnable);

  bool Do() override;
  bool GetBoundingBox(VistaBoundingBox& bb) override;

 private:
  void initData();
  void updateShader();

  std::shared_ptr<Plugin::Settings> mPluginSettings;
  std::unique_ptr<VistaTexture>     mCloudTexture;
  std::string                       mCloudTextureFile;
  float                             mCloudHeight    = 0.001F;
  bool                              mUseClouds      = false;
  glm::dvec3                        mRadii          = glm::dvec3(1.0, 1.0, 1.0);
  glm::dmat4                        mWorldTransform = glm::dmat4(1.0);

  std::shared_ptr<cs::graphics::ShadowMap> mShadowMap;
  std::shared_ptr<cs::graphics::HDRBuffer> mHDRBuffer;

  VistaGLSLShader        mAtmoShader;
  VistaVertexArrayObject mQuadVAO;
  VistaBufferObject      mQuadVBO;

  struct GBufferData {
    std::unique_ptr<VistaTexture> mDepthBuffer;
    std::unique_ptr<VistaTexture> mColorBuffer;
  };

  std::unordered_map<VistaViewport*, GBufferData> mGBufferData;

  bool      mShaderDirty       = true;
  bool      mDrawSun           = true;
  bool      mDrawWater         = false;
  float     mWaterLevel        = 0.0F;
  float     mAmbientBrightness = 0.2F;
  float     mAtmosphereHeight  = 1.0F;
  int       mPrimaryRaySteps   = 15;
  int       mSecondaryRaySteps = 4;
  float     mSunIntensity      = 1.F;
  glm::vec3 mSunDirection      = glm::vec3(1, 0, 0);

  float     mMieHeight     = 0.0F;
  glm::vec3 mMieScattering = glm::vec3(1, 1, 1);
  float     mMieAnisotropy = 0.0F;

  float     mRayleighHeight     = 0.0F;
  glm::vec3 mRayleighScattering = glm::vec3(1, 1, 1);
  float     mRayleighAnisotropy = 0.0F;

  float mApproximateBrightness = 0.0F;

  bool  mUseLinearDepthBuffer = false;
  bool  mUseToneMapping       = true;
  float mExposure             = 0.6F;
  float mGamma                = 2.2F;

  struct {
    uint32_t sunIntensity      = 0;
    uint32_t sunDir            = 0;
    uint32_t farClip           = 0;
    uint32_t waterLevel        = 0;
    uint32_t ambientBrightness = 0;
    uint32_t depthBuffer       = 0;
    uint32_t colorBuffer       = 0;
    uint32_t cloudTexture      = 0;
    uint32_t cloudAltitude     = 0;
    uint32_t shadowCascades    = 0;

    std::array<uint32_t, 5> shadowMaps{};
    std::array<uint32_t, 5> shadowProjectionMatrices{};

    uint32_t inverseModelViewMatrix           = 0;
    uint32_t inverseModelViewProjectionMatrix = 0;
    uint32_t inverseProjectionMatrix          = 0;
    uint32_t modelViewMatrix                  = 0;
  } mUniforms;

  static const char* cAtmosphereVert;
  static const char* cAtmosphereFrag0;
  static const char* cAtmosphereFrag1;
};

} // namespace csp::atmospheres

#endif // CSP_ATMOSPHERE_RENDERER_HPP
