////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CS_GRAPHICS_CLEAR_HDRBUFFER_NODE_HPP
#define CS_GRAPHICS_CLEAR_HDRBUFFER_NODE_HPP

#include "cs_graphics_export.hpp"

#include <VistaKernel/GraphicsManager/VistaOpenGLDraw.h>
#include <memory>

namespace cs::graphics {

class HDRBuffer;

/// This node clears and binds the given HDRBuffer when its Do() method is called.
class CS_GRAPHICS_EXPORT ClearHDRBufferNode : public IVistaOpenGLDraw {
 public:
  explicit ClearHDRBufferNode(std::shared_ptr<HDRBuffer> hdrBuffer);

  bool Do() override;
  bool GetBoundingBox(VistaBoundingBox& oBoundingBox) override;

 private:
  std::shared_ptr<HDRBuffer> mHDRBuffer;
};

} // namespace cs::graphics

#endif // CS_GRAPHICS_CLEAR_HDRBUFFER_NODE_HPP
