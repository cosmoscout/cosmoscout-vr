////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CS_GRAPHICS_MOUSE_RAY_HPP
#define CS_GRAPHICS_MOUSE_RAY_HPP

#include "cs_graphics_export.hpp"

#include <memory>

#include <VistaKernel/GraphicsManager/VistaOpenGLDraw.h>
#include <VistaKernel/GraphicsManager/VistaOpenGLNode.h>
#include <VistaOGLExt/VistaBufferObject.h>
#include <VistaOGLExt/VistaGLSLShader.h>
#include <VistaOGLExt/VistaVertexArrayObject.h>

class VistaTransformNode;
class VistaOpenGLNode;

namespace cs::graphics {

/// A ray that shoots from the mouse into space? DocTODO
class CS_GRAPHICS_EXPORT MouseRay : public IVistaOpenGLDraw {
 public:
  MouseRay();

  MouseRay(MouseRay const& other) = delete;
  MouseRay(MouseRay&& other)      = delete;

  MouseRay& operator=(MouseRay const& other) = delete;
  MouseRay& operator=(MouseRay&& other)      = delete;

  ~MouseRay() override = default;

  bool Do() override;
  bool GetBoundingBox(VistaBoundingBox& bb) override;

 private:
  std::unique_ptr<VistaTransformNode> mRayTransform;
  std::unique_ptr<VistaOpenGLNode>    mMouseRayNode;

  VistaGLSLShader        mShader;
  VistaVertexArrayObject mRayVAO;
  VistaBufferObject      mRayVBO;
  VistaBufferObject      mRayIBO;

  struct {
    uint32_t modelViewMatrix  = 0;
    uint32_t projectionMatrix = 0;
  } mUniforms;
};

} // namespace cs::graphics

#endif // CS_GRAPHICS_MOUSE_RAY_HPP
