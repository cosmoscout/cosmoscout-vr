////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSP_SHARAD_HPP
#define CSP_SHARAD_HPP

#include "../../../src/cs-core/GraphicsEngine.hpp"
#include "../../../src/cs-core/Settings.hpp"
#include "../../../src/cs-core/SolarSystem.hpp"

#include <VistaKernel/GraphicsManager/VistaOpenGLDraw.h>
#include <VistaKernel/GraphicsManager/VistaSceneGraph.h>
#include <VistaOGLExt/VistaBufferObject.h>
#include <VistaOGLExt/VistaGLSLShader.h>
#include <VistaOGLExt/VistaVertexArrayObject.h>

#include <memory>

class VistaTexture;

namespace csp::sharad {

/// Renders a single SHARAD image.
class Sharad : public IVistaOpenGLDraw {
 public:
  Sharad(std::shared_ptr<cs::core::Settings>    settings,
      std::shared_ptr<cs::core::GraphicsEngine> graphicsEngine,
      std::shared_ptr<cs::core::SolarSystem> solarSystem, std::string objectName,
      std::string const& sTiffFile, std::string const& sTabFile);

  Sharad(Sharad const& other) = delete;
  Sharad(Sharad&& other)      = delete;

  Sharad& operator=(Sharad const& other) = delete;
  Sharad& operator=(Sharad&& other)      = delete;

  double getStartTime() const;

  void update(double tTime, double sceneScale);

  bool Do() override;
  bool GetBoundingBox(VistaBoundingBox& bb) override;

 private:
  std::shared_ptr<cs::core::Settings>       mSettings;
  std::shared_ptr<cs::core::GraphicsEngine> mGraphicsEngine;
  std::shared_ptr<cs::core::SolarSystem>    mSolarSystem;
  std::unique_ptr<VistaTexture>             mTexture;

  std::string mObjectName;
  double      mStartTime;

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
  } mUniforms;

  int    mSamples;
  double mCurrTime   = -1.0;
  double mSceneScale = -1.0;

  static const char* VERT;
  static const char* FRAG;
};

} // namespace csp::sharad

#endif // CSP_SHARAD_HPP
