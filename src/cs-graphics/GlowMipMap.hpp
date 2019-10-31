////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CS_GRAPHICS_GLOW_MIPMAP_HPP
#define CS_GRAPHICS_GLOW_MIPMAP_HPP

#include "HDRBuffer.hpp"

#include <VistaOGLExt/VistaTexture.h>
#include <memory>

namespace cs::graphics {

class CS_GRAPHICS_EXPORT GlowMipMap : public VistaTexture {
 public:
  GlowMipMap(int hdrBufferWidth, int hdrBufferHeight);
  virtual ~GlowMipMap();

  void update(VistaTexture* hdrBufferComposite);

 private:
  GLuint mComputeProgram  = 0;
  int    mMaxLevels       = 0;
  int    mHDRBufferWidth  = 0;
  int    mHDRBufferHeight = 0;

  VistaTexture* mTemporaryTarget = nullptr;

  static const std::string sGlowShader;
};

} // namespace cs::graphics

#endif // CS_GRAPHICS_GLOW_MIPMAP_HPP
