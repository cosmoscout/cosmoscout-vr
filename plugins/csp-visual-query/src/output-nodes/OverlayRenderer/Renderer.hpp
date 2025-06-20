////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSP_VISUAL_QUERY_RENDERER_HPP
#define CSP_VISUAL_QUERY_RENDERER_HPP

#include "../../types/types.hpp"

#include <VistaKernel/GraphicsManager/VistaOpenGLDraw.h>
#include <VistaOGLExt/VistaGLSLShader.h>
#include <VistaOGLExt/VistaTexture.h>

#include <memory>
#include <string>

// FORWARD DEFINITIONS
class VistaViewport;
class VistaGLSLShader;
class VistaOpenGLNode;
class VistaTexture;

namespace cs::core {
class SolarSystem;
class Settings;
} // namespace cs::core

namespace csp::visualquery {

class Renderer final : public IVistaOpenGLDraw {
 public:
  Renderer(std::shared_ptr<cs::core::SolarSystem> solarSystem,
      std::shared_ptr<cs::core::Settings>         settings);
  ~Renderer() override;
  bool Do() override;
  bool GetBoundingBox(VistaBoundingBox& bb) override;

  void        setData(std::shared_ptr<Image2D> const& image);
  void        setLUT(std::vector<glm::vec4> const& lut);
  void        setMinMax(glm::vec2 const& minMax);
  void        setOpacity(float opacity);
  float       getOpacity() const;
  void        setObject(std::string objectName);
  std::string getObject() const;

 private:
  std::string                            mObjectName;
  std::shared_ptr<cs::core::SolarSystem> mSolarSystem;
  std::shared_ptr<cs::core::Settings>    mSettings;

  std::unique_ptr<VistaOpenGLNode> mGLNode;

  /// Vista GLSL shader object used for rendering
  VistaGLSLShader mShader;

  VistaTexture mTexture;
  bool         mHasTexture;

  VistaTexture mLUT;
  bool         mHasLUT;

  glm::vec2 mMinMax  = glm::vec2(0, 1);
  float     mOpacity = 1.0f;
  size_t    mNumScalars;

  /// Store one buffer per viewport
  std::map<VistaViewport*, VistaTexture> mDepthBufferData;

  /// Lower Corner of the bounding volume for the planet.
  glm::vec3 mMinBounds;

  /// Upper Corner of the bounding volume for the planet.
  glm::vec3 mMaxBounds;

  csl::ogc::Bounds2D mBounds;

  bool mShaderDirty = true;

  /// Code for the geometry shader
  static const std::string SURFACE_GEOM;
  /// Code for the vertex shader
  static const std::string SURFACE_VERT;
  /// Code for the fragment shader
  static const std::string SURFACE_FRAG;
};

} // namespace csp::visualquery

#endif // CSP_VISUAL_QUERY_RENDERER_HPP
