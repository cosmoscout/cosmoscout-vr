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

/// The LuminanceMipMap is a texture with full mipmap levels which are used to calculate the total
/// and maximum luminance of the current scene by parallel reduction. It's a 32bit RG texture of
/// half the given width and height.
class CS_GRAPHICS_EXPORT LuminanceMipMap : public VistaTexture {
 public:
  LuminanceMipMap(uint32_t hdrBufferSamples, int hdrBufferWidth, int hdrBufferHeight);
  ~LuminanceMipMap() override;

  LuminanceMipMap(LuminanceMipMap const& other) = delete;
  LuminanceMipMap(LuminanceMipMap&& other)      = delete;

  LuminanceMipMap& operator=(LuminanceMipMap const& other) = delete;
  LuminanceMipMap& operator=(LuminanceMipMap&& other) = delete;

  /// Perform the parallel reduction of luminance values. This is a costly operation and should only
  /// be called once a frame.
  void update(VistaTexture* hdrBufferComposite);

  /// Returns true once data has been retrieved from the GPU. This will be one frame after the first
  /// call to update().
  bool getIsDataAvailable() const;

  /// Get the results of the last but one call to update(). The data is read back from the GPU one
  /// frame after the computation in order to reduce synchronization requirements. In order to get
  /// the average luminance, you have to divide getLastTotalLuminance() by (hdrBufferWidth *
  /// hdrBufferHeight).
  float getLastTotalLuminance() const;
  float getLastMaximumLuminance() const;

 private:
  GLuint   mPBO                  = 0;
  GLuint   mComputeProgram       = 0;
  uint32_t mHDRBufferSamples     = 0;
  float    mLastTotalLuminance   = 0.F;
  float    mLastMaximumLuminance = 0.F;
  int      mMaxLevels            = 0;
  int      mHDRBufferWidth       = 0;
  int      mHDRBufferHeight      = 0;
  bool     mDataAvailable        = false;

  struct {
    uint32_t level = 0;
  } mUniforms;
};

} // namespace cs::graphics

#endif // CS_GRAPHICS_LUMINANCE_MIPMAP_HPP
