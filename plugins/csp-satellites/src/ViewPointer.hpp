////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSP_SATELLITES_VIEWPOINTER_HPP
#define CSP_SATELLITES_VIEWPOINTER_HPP

#include "Plugin.hpp"

#include "../../../src/cs-scene/CelestialObject.hpp"

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

namespace csp::satellites {

class ViewPointer : public IVistaOpenGLDraw {
 public:
  ViewPointer(std::shared_ptr<cs::core::SolarSystem> solarSystem, std::string const& anchorName);

  ViewPointer(ViewPointer const& other) = delete;
  ViewPointer(ViewPointer&& other)      = default;

  ViewPointer& operator=(ViewPointer const& other) = delete;
  ViewPointer& operator=(ViewPointer&& other) = delete;

  ~ViewPointer();

  /// Configures the internal renderer according to the given values.
  void configure(Plugin::Settings const&);

  /// Updates the offset of the grid according to the current settings
  void update();

  bool Do() override;
  bool GetBoundingBox(VistaBoundingBox& bb) override;

 private:
  std::shared_ptr<cs::core::SolarSystem> mSolarSystem;
  std::string                            mAnchorName;
  std::string                            mBodyName = "Earth";

  std::unique_ptr<VistaOpenGLNode> mGLNode;

  VistaGLSLShader        mShader;
  VistaVertexArrayObject mVAO;
  VistaBufferObject      mVBO;

  struct {
    uint32_t modelViewMatrix  = 0;
    uint32_t projectionMatrix = 0;
    uint32_t alpha            = 0;
    uint32_t color            = 0;
  } mUniforms;

  static const char* VERT_SHADER;
  static const char* FRAG_SHADER;
}; // class ViewPointer
} // namespace csp::satellites

#endif // CSP_SATELLITES_VIEWPOINTER_HPP
