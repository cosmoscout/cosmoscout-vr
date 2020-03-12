////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CS_GRAPHICS_LUMINANCE_MIPMAP_HPP
#define CS_GRAPHICS_LUMINANCE_MIPMAP_HPP

#include "HDRBuffer.hpp"

#include <VistaOGLExt/VistaTexture.h>
#include <memory>

namespace cs::graphics {

/// The LuminanceMipMap is a texture with full mipmap levels which are used to calculate the average
/// and maximum luminance of the current scene by parallel reduction. It's a 32bit RG texture of
/// half the given width and height.
class CS_GRAPHICS_EXPORT LuminanceMipMap : public VistaTexture {
 public:
  LuminanceMipMap(int hdrBufferWidth, int hdrBufferHeight);
  virtual ~LuminanceMipMap();

  void update(VistaTexture* hdrBufferComposite);

  /// Returns true once data has been retrieved from the GPU. This will be one frame after the first
  /// call to update().
  bool getIsDataAvailable() const;

  /// Get the
  float getLastTotalLuminance() const;
  float getLastMaximumLuminance() const;

 private:
  GLuint mPBO                  = 0;
  GLuint mComputeProgram       = 0;
  float  mLastTotalLuminance   = 0.f;
  float  mLastMaximumLuminance = 0.f;
  int    mMaxLevels            = 0;
  int    mHDRBufferWidth       = 0;
  int    mHDRBufferHeight      = 0;
  bool   mDataAvailable        = false;

  static const std::string sComputeAverage;
};

} // namespace cs::graphics

#endif // CS_GRAPHICS_LUMINANCE_MIPMAP_HPP
