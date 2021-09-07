////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CSP_VR_ACCESSIBILITY_FOVVIGNETTE_HPP
#define CSP_VR_ACCESSIBILITY_FOVVIGNETTE_HPP

#include "Plugin.hpp"

#include "../../../src/cs-scene/CelestialObject.hpp"
#include "../../../src/cs-utils/AnimatedValue.hpp"

#include <VistaKernel/DisplayManager/VistaViewport.h>
#include <VistaKernel/GraphicsManager/VistaOpenGLDraw.h>
#include <VistaKernel/GraphicsManager/VistaOpenGLNode.h>
#include <VistaKernel/GraphicsManager/VistaTransformNode.h>
#include <VistaOGLExt/VistaBufferObject.h>
#include <VistaOGLExt/VistaGLSLShader.h>
#include <VistaOGLExt/VistaTexture.h>
#include <VistaOGLExt/VistaVertexArrayObject.h>

namespace cs::core {
class Settings;
class SolarSystem;
} // namespace cs::core

namespace csp::vraccessibility {

/// The FoV Vignette draws a vignette when the observer is moving.
/// The Vignette is split into 4 different Options: dynamical or static vignetting, and vertical or
/// circular vignette.
class FovVignette : public IVistaOpenGLDraw {
 public:
  FovVignette(std::shared_ptr<cs::core::SolarSystem> solarSystem,
      Plugin::Settings::Vignette&                    vignetteSettings);

  FovVignette(FovVignette const& other) = delete;
  FovVignette(FovVignette&& other)      = default;

  FovVignette& operator=(FovVignette const& other) = delete;
  FovVignette& operator=(FovVignette&& other) = delete;

  ~FovVignette();

  /// Configures the internal renderer according to the given values.
  void configure(Plugin::Settings::Vignette& vignetteSettings);

  /// Updates the variables for the Shaders
  void updateDynamicRadiusVignette();
  void updateFadeAnimatedVignette();

  bool Do() override;
  bool GetBoundingBox(VistaBoundingBox& bb) override;

 private:
  float getNewRadius(float innerOuterRadius, float normVelocity, float lastRadius, double dT);
  double getNow();

  std::shared_ptr<cs::core::Settings>    mSettings;
  std::shared_ptr<cs::core::SolarSystem> mSolarSystem;

  std::unique_ptr<VistaOpenGLNode> mGLNode;

  cs::utils::AnimatedValue<float> mFadeAnimation;
  double                          mLastChange = std::numeric_limits<double>::max();
  bool                            mIsMoving   = false;

  glm::vec2                                                       mCurrentRadii = glm::vec2(1.F);
  glm::vec2                                                       mLastRadii = glm::vec2(1.4142F);
  std::chrono::time_point<std::chrono::high_resolution_clock> mLastTime;
  float                                                       mNormalizedVelocity;

  Plugin::Settings::Vignette& mVignetteSettings;
  VistaGLSLShader             mShaderFade;
  VistaGLSLShader             mShaderDynRad;
  VistaGLSLShader             mShaderFadeVertOnly;
  VistaGLSLShader             mShaderDynRadVertOnly;
  VistaVertexArrayObject      mVAO;
  VistaBufferObject           mVBO;

  struct {
    struct {
      uint32_t aspect       = 0;
      uint32_t normVelocity = 0;
      uint32_t color        = 0;
      uint32_t radii        = 0;
      uint32_t debug        = 0;
    } dynamic, dynamicVertical;

    struct {
      uint32_t aspect   = 0;
      uint32_t fade     = 0;
      uint32_t color    = 0;
      uint32_t radii    = 0;
      uint32_t debug    = 0;
    } fade, fadeVertical;

  } mUniforms;

  static const char* VERT_SHADER;
  static const char* FRAG_SHADER_FADE;
  static const char* FRAG_SHADER_DYNRAD;
  static const char* FRAG_SHADER_FADE_VERTONLY;
  static const char* FRAG_SHADER_DYNRAD_VERTONLY;
}; // class FloorGrid
} // namespace csp::vraccessibility

#endif // CSP_VR_ACCESSIBILITY_FOVVIGNETTE_HPP
