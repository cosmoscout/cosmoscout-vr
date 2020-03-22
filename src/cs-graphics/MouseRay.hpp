////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

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

  bool Do() override;
  bool GetBoundingBox(VistaBoundingBox& bb) override;

 private:
  std::unique_ptr<VistaTransformNode> mRayTransform;
  std::unique_ptr<VistaOpenGLNode>    mMouseRayNode;

  VistaGLSLShader        mShader;
  VistaVertexArrayObject mRayVAO;
  VistaBufferObject      mRayVBO;
  VistaBufferObject      mRayIBO;

  static const std::string SHADER_VERT;
  static const std::string SHADER_FRAG;
};

} // namespace cs::graphics

#endif // CS_GRAPHICS_MOUSE_RAY_HPP
