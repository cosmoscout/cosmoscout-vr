////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSP_VISUAL_QUERY_RENDERER_HPP
#define CSP_VISUAL_QUERY_RENDERER_HPP

#include <VistaOGLExt/VistaGLSLShader.h>
#include <VistaOGLExt/VistaTexture.h>

#include <glm/vec3.hpp>

#include <map>
#include <memory>
#include <string>

class VistaOpenGLNode;
class VistaViewport;

namespace csp::visualquery {

class Renderer final : public IVistaOpenGLDraw {
 public:
  explicit Renderer(std::string objectName, std::shared_ptr<cs::core::SolarSystem> solarSystem);
  ~Renderer() override;
  bool Do() override;
  bool GetBoundingBox(VistaBoundingBox& bb) override;

 private:
  std::string                            mObjectName;
  std::shared_ptr<cs::core::SolarSystem> mSolarSystem;

  std::unique_ptr<VistaOpenGLNode> mGLNode;

  /// Vista GLSL shader object used for rendering
  VistaGLSLShader mShader;

  VistaTexture mTexture;

  /// Store one buffer per viewport
  std::map<VistaViewport*, VistaTexture> mDepthBufferData;

  /// Lower Corner of the bounding volume for the planet.
  glm::vec3 mMinBounds;

  /// Upper Corner of the bounding volume for the planet.
  glm::vec3 mMaxBounds;

  bool mShaderDirty        = true;

  /// Code for the geometry shader
  static const std::string SURFACE_GEOM;
  /// Code for the vertex shader
  static const std::string SURFACE_VERT;
  /// Code for the fragment shader
  static const std::string SURFACE_FRAG;
};

} // namespace csp::visualquery

#endif // CSP_VISUAL_QUERY_RENDERER_HPP
