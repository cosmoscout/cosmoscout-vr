////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CS_GRAPHICS_GLOW_MIPMAP_HPP
#define CS_GRAPHICS_GLOW_MIPMAP_HPP

#include "HDRBuffer.hpp"

#include <VistaOGLExt/VistaTexture.h>

namespace cs::graphics {

/// This is a 32bit RGBA texture of half the given width and height with full mipmap levels.
/// Whenever update() is called, all mipmap levels are updated using compute shaders in several
/// passes to contain a blurred version of the given texture. The blur radius increases with the
/// mipmap level.
class CS_GRAPHICS_EXPORT GlowMipMap : public VistaTexture {
 public:
  GlowMipMap(uint32_t hdrBufferSamples, int hdrBufferWidth, int hdrBufferHeight);
  ~GlowMipMap() override;

  GlowMipMap(GlowMipMap const& other) = delete;
  GlowMipMap(GlowMipMap&& other)      = delete;

  GlowMipMap& operator=(GlowMipMap const& other) = delete;
  GlowMipMap& operator=(GlowMipMap&& other) = delete;

  /// Perform the glow calculation by parallel reduction of the HDR values. This is a costly
  /// operation and should only be called once a frame.
  void update(VistaTexture* hdrBufferComposite);

 private:
  GLuint   mComputeProgram   = 0;
  uint32_t mHDRBufferSamples = 0;
  int      mMaxLevels        = 0;
  int      mHDRBufferWidth   = 0;
  int      mHDRBufferHeight  = 0;

  struct {
    uint32_t level = 0;
    uint32_t pass  = 0;
  } mUniforms;

  VistaTexture* mTemporaryTarget = nullptr;
};

} // namespace cs::graphics

#endif // CS_GRAPHICS_GLOW_MIPMAP_HPP
