////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CS_GRAPHICS_HDRBUFFER_HPP
#define CS_GRAPHICS_HDRBUFFER_HPP

#include "cs_graphics_export.hpp"

#include <array>
#include <cstdint>
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
  explicit HDRBuffer(uint32_t multiSamples, bool highPrecision = true);

  HDRBuffer(HDRBuffer const& other) = delete;
  HDRBuffer(HDRBuffer&& other)      = delete;

  HDRBuffer& operator=(HDRBuffer const& other) = delete;
  HDRBuffer& operator=(HDRBuffer&& other) = delete;

  virtual ~HDRBuffer();

  /// Returns the number of multi-samples used by this HDRBuffer. You should check this number
  /// before reading from the attachments. See getDepthAttachment() and getCurrentReadAttachment()
  /// further below.
  uint32_t getMultiSamples() const;

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
  void calculateLuminance();

  // Get the results of the last but one call to calculateLuminance(). The data is read back from
  // the GPU one
  /// frame after the computation in order to reduce synchronization requirements. In order to get
  /// the average luminance, you have to divide getLastTotalLuminance() by (hdrBufferWidth *
  /// hdrBufferHeight).
  float getTotalLuminance() const;
  float getMaximumLuminance() const;

  /// Update and access the GlowMipMap.
  void          updateGlowMipMap();
  VistaTexture* getGlowMipMap() const;

  /// Returns the depth attachment for the currently rendered viewport. Be aware, that this can be
  /// texture with the target GL_TEXTURE_2D_MULTISAMPLE if getMultiSamples() > 0.
  VistaTexture* getDepthAttachment() const;

  /// Returns the color attachment which is currently bound for writing for the currently rendered
  /// viewport.
  VistaTexture* getCurrentWriteAttachment() const;

  /// Returns the color attachment which is currently bound for reading for the currently rendered
  /// viewport. Be aware, that this can be texture with the target GL_TEXTURE_2D_MULTISAMPLE if
  /// getMultiSamples() > 0.
  VistaTexture* getCurrentReadAttachment() const;

  /// Helper methods to access the size and position of the viewports we are currently rendering to.
  static std::array<int, 2> getCurrentViewPortSize();
  static std::array<int, 2> getCurrentViewPortPos();

 private:
  // There is one of these structs for each viewport. That means, we have a separate framebuffer
  // object, GlowMipMap and LuminanceMipMap for each viewport. This is mainly because viewports
  // often have different sizes.
  struct HDRBufferData {
    VistaFramebufferObj*         mFBO{};
    std::array<VistaTexture*, 2> mColorAttachments{};
    VistaTexture*                mDepthAttachment{};
    LuminanceMipMap*             mLuminanceMipMap{};
    GlowMipMap*                  mGlowMipMap{};

    // Stores the original viewport position and size.
    std::array<int, 4> mCachedViewport{};
    int                mWidth                 = 0;
    int                mHeight                = 0;
    int                mCompositePinpongState = 0;
    bool               mIsBound               = false;
  };

  // Helper methods to retrieve the current HDRBufferData struct based on the viewport we are
  // currently rendering to.
  HDRBufferData&       getCurrentHDRBuffer();
  HDRBufferData const& getCurrentHDRBuffer() const;

  std::unordered_map<VistaViewport*, HDRBufferData> mHDRBufferData;
  float                                             mTotalLuminance   = 1.F;
  float                                             mMaximumLuminance = 1.F;

  const uint32_t mMultiSamples;
  const bool     mHighPrecision;
};

} // namespace cs::graphics

#endif // CS_GRAPHICS_HDRBUFFER_HPP
