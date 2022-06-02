////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CS_GRAPHICS_SETUP_GL_NODE_HPP
#define CS_GRAPHICS_SETUP_GL_NODE_HPP

#include "cs_graphics_export.hpp"

#include <VistaKernel/GraphicsManager/VistaOpenGLDraw.h>

namespace cs::graphics {

class CS_GRAPHICS_EXPORT SetupGLNode : public IVistaOpenGLDraw {
 public:
  bool Do() override;
  bool GetBoundingBox(VistaBoundingBox& oBoundingBox) override;
};

} // namespace cs::graphics

#endif // CS_GRAPHICS_SETUP_GL_NODE_HPP
