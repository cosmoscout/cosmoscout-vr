////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CS_GRAPHICS_SETUP_GL_NODE_HPP
#define CS_GRAPHICS_SETUP_GL_NODE_HPP

#include "cs_graphics_export.hpp"

#include <VistaKernel/GraphicsManager/VistaOpenGLDraw.h>

namespace cs::graphics {

/// The GraphicsEngine will put one instance of this class into the scenegraph. There, it is
/// responsible to initialize the GL state to the defaults required by CosmoScout VR. This includes:
/// - Setting depth func to GL_GEQUAL (required for the reverse infinite projection)
/// - Enabling back-face culling
class CS_GRAPHICS_EXPORT SetupGLNode : public IVistaOpenGLDraw {
 public:
  bool Do() override;
  bool GetBoundingBox(VistaBoundingBox& oBoundingBox) override;
};

} // namespace cs::graphics

#endif // CS_GRAPHICS_SETUP_GL_NODE_HPP
