////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CS_GRAPHICS_HDRBUFFER_HPP
#define CS_GRAPHICS_HDRBUFFER_HPP

#include "cs_graphics_export.hpp"

#include <VistaOGLExt/VistaTexture.h>

#include <array>
#include <unordered_map>
#include <vector>

class VistaFramebufferObj;
class VistaTexture;
class VistaViewport;
class VistaGLSLShader;

namespace cs::graphics {

class LuminanceMipMap;
class GlowMipMap;

/// The HDRBuffer is used as render target when HDR rendering is enabled. It contains an framebuffer
/// object for each viewport. Each framebuffer object has two color attachments containing luminance
/// values (which can be used in a ping-pong fashion) and a depth attachment. It also contains a
/// LuminanceMipMap to compute the average brightness for auto-exposure and a GlowMipMap for a
/// glare-effect.
class CS_GRAPHICS_EXPORT HDRBuffer {
 public:
  /// When highPrecision is set to false, only 16bit color buffers are used.
  HDRBuffer(bool highPrecision = true);
  virtual ~HDRBuffer();

  /// Binds DEPTH and one ping-pong target for writing.
  void bind();

  /// Subsequent calls will draw to the backbuffer again.
  void unbind();

  /// Subsequent calls to bind() will use the ping-pong targets the other way around.
  void doPingPong();

  /// Clears all attachments, that is DEPTH to 1.0, and HDR_0 and HDR_1 to vec3(0).
  void clear();

  /// Calculate the scene's total and maximum luminance. The results can be retrieved with
  /// getTotalLuminance() and getMaximumLuminance().
  void  calculateLuminance();
  float getTotalLuminance() const;
  float getMaximumLuminance() const;

  /// Update and access the GlowMipMap.
  void          updateGlowMipMap();
  VistaTexture* getGlowMipMap() const;

  /// Returns the depth attachment for the currently rendered viewport.
  VistaTexture* getDepthAttachment() const;

  /// Returns the color attachment which is currently bound for writing for the currently rendered
  /// viewport.
  VistaTexture* getCurrentWriteAttachment() const;

  /// Returns the color attachment which is currently bound for reading for the currently rendered
  /// viewport.
  VistaTexture* getCurrentReadAttachment() const;

  /// Helper methods to access the size and position of the viewports we are currently rendering to.
  std::array<int, 2> getCurrentViewPortSize() const;
  std::array<int, 2> getCurrentViewPortPos() const;

 private:
  struct HDRBufferData {
    VistaFramebufferObj*         mFBO              = nullptr;
    std::array<VistaTexture*, 2> mColorAttachments = {{nullptr, nullptr}};
    VistaTexture*                mDepthAttachment  = nullptr;
    LuminanceMipMap*             mLuminanceMipMap  = nullptr;
    GlowMipMap*                  mGlowMipMap       = nullptr;

    int  mCachedViewport[4];
    int  mWidth                 = 0;
    int  mHeight                = 0;
    int  mCompositePinpongState = 0;
    bool mIsBound               = false;
  };

  HDRBufferData&       getCurrentHDRBuffer();
  HDRBufferData const& getCurrentHDRBuffer() const;

  std::unordered_map<VistaViewport*, HDRBufferData> mHDRBufferData;
  float                                             mTotalLuminance   = 1.f;
  float                                             mMaximumLuminance = 1.f;

  const float mHighPrecision;
};

} // namespace cs::graphics

#endif // CS_GRAPHICS_HDRBUFFER_HPP
