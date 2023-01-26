////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CS_GRAPHICS_SETUP_GL_NODE_HPP
#define CS_GRAPHICS_SETUP_GL_NODE_HPP

#include "cs_graphics_export.hpp"
#include "../cs-core/Settings.hpp"

#include <VistaKernel/GraphicsManager/VistaOpenGLDraw.h>

namespace cs::graphics {

class CS_GRAPHICS_EXPORT SetupGLNode : public IVistaOpenGLDraw {
 public:
  //The class needs access to the settings in order to read the log level
  SetupGLNode(std::shared_ptr<cs::core::Settings> settings);
  
  bool Do() override;
  bool GetBoundingBox(VistaBoundingBox& oBoundingBox) override;

private:
  std::shared_ptr<cs::core::Settings>     mSettings;
};

} // namespace cs::graphics

#endif // CS_GRAPHICS_SETUP_GL_NODE_HPP
