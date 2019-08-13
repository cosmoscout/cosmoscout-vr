////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CS_GRAPHICS_CLEAR_HDRBUFFER_NODE_HPP
#define CS_GRAPHICS_CLEAR_HDRBUFFER_NODE_HPP

#include "cs_graphics_export.hpp"

#include <VistaKernel/GraphicsManager/VistaOpenGLDraw.h>
#include <memory>

namespace cs::graphics {

class HDRBuffer;

class CS_GRAPHICS_EXPORT ClearHDRBufferNode : public IVistaOpenGLDraw {
 public:
  ClearHDRBufferNode(std::shared_ptr<HDRBuffer> const& hdrBuffer);
  virtual ~ClearHDRBufferNode();

  virtual bool Do() override;
  virtual bool GetBoundingBox(VistaBoundingBox& bb) override;

 private:
  std::shared_ptr<HDRBuffer> mHDRBuffer;
};

} // namespace cs::graphics

#endif // CS_GRAPHICS_CLEAR_HDRBUFFER_NODE_HPP
