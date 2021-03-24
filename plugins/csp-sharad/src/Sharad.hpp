////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CSP_SHARAD_HPP
#define CSP_SHARAD_HPP

#include "../../../src/cs-core/Settings.hpp"
#include "../../../src/cs-core/SolarSystem.hpp"
#include "../../../src/cs-scene/CelestialObject.hpp"

#include <VistaKernel/GraphicsManager/VistaOpenGLDraw.h>
#include <VistaKernel/GraphicsManager/VistaSceneGraph.h>
#include <VistaOGLExt/VistaBufferObject.h>
#include <VistaOGLExt/VistaGLSLShader.h>
#include <VistaOGLExt/VistaVertexArrayObject.h>

#include <memory>

class VistaTexture;

namespace csp::sharad {

/// Renders a single SHARAD image.
class Sharad : public cs::scene::CelestialObject, public IVistaOpenGLDraw {
 public:
  Sharad(std::shared_ptr<cs::core::Settings> settings, std::string const& anchorName,
      std::string const& sTiffFile, std::string const& sTabFile);

  Sharad(Sharad const& other) = delete;
  Sharad(Sharad&& other)      = delete;

  Sharad& operator=(Sharad const& other) = delete;
  Sharad& operator=(Sharad&& other) = delete;

  ~Sharad() override;

  void update(double tTime, cs::scene::CelestialObserver const& oObs) override;

  bool Do() override;
  bool GetBoundingBox(VistaBoundingBox& bb) override;

 private:
  class FramebufferCallback : public IVistaOpenGLDraw {
   public:
    explicit FramebufferCallback(VistaTexture* pDepthBuffer);

    bool Do() override;
    bool GetBoundingBox(VistaBoundingBox& /*bb*/) override {
      return true;
    }

   private:
    VistaTexture* mDepthBuffer;
  };

  static std::unique_ptr<VistaTexture>        mDepthBuffer;
  static std::unique_ptr<FramebufferCallback> mPreCallback;
  static std::unique_ptr<VistaOpenGLNode>     mPreCallbackNode;
  static int                                  mInstanceCount;

  std::shared_ptr<cs::core::Settings> mSettings;
  std::unique_ptr<VistaTexture>       mTexture;

  VistaGLSLShader        mShader;
  VistaVertexArrayObject mVAO;
  VistaBufferObject      mVBO;

  struct {
    uint32_t modelViewMatrix  = 0;
    uint32_t projectionMatrix = 0;
    uint32_t viewportPosition = 0;
    uint32_t sharadTexture    = 0;
    uint32_t depthBuffer      = 0;
    uint32_t sceneScale       = 0;
    uint32_t heightScale      = 0;
    uint32_t radii            = 0;
    uint32_t time             = 0;
    uint32_t farClip          = 0;
  } mUniforms;

  int    mSamples;
  double mCurrTime   = -1.0;
  double mSceneScale = -1.0;

  static const char* VERT;
  static const char* FRAG;
};

} // namespace csp::sharad

#endif // CSP_SHARAD_HPP
