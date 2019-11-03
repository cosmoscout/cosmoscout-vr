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

enum class ExposureMeteringMode { AVERAGE = 0 };

/**
 * Rendering with VistaDeferredPBRExport
VistaDeferredPBR involves several required steps
 * - ClearHDRBufferNode::Do()   Should be called quite early, this will clear and bind the
HDRBuffer.
 * - Draw opaque geometry     The HDRBuffer has several attachements your shaders are supposed to
 *                            write to:
 *                            gl_FragDepth = length(viewSpacePosition) / farClip;
 *                            layout(location = 0) out vec3 oNormal                       3x16 bit
 *                            layout(location = 1) out vec3 oAlbedo                       3x8  bit
 *                            layout(location = 2) out vec3 oEmissivity;                  3x16 bit
 *                            layout(location = 3) out vec3 oRoughnessMetalnessOcclusion; 3x8  bit
 * - HDRBufferResolveNode::Do() This will do physically based shading based on the HDRBuffer, light
 *                            information and an environment map and store the result in another
 *                            3x16 bit target
 * - Apply HDR effects        OpenGLNodes may use the HDRBuffer interface to apply effects such as
 *                            bloom or lensflares or tone-mapping. The internal render target will
 *                            be used in a ping-pong fashion. The last effect should be configured
 *                            to write to the backbuffer. HDRBuffer attachments will be bound to
 *                            texture units and can be accessed like this:
 *                            layout(binding = 0) uniform sampler2D uDepth;
 *                            layout(binding = 1) uniform sampler2D uNormal;
 *                            layout(binding = 2) uniform sampler2D uAlbedo;
 *                            layout(binding = 3) uniform sampler2D uEmissivity;
 *                            layout(binding = 4) uniform sampler2D uRoughnessMetalnessOcclusion;
 *                            layout(binding = 5) uniform sampler2D uComposite;
 */
class CS_GRAPHICS_EXPORT HDRBuffer {
 public:
  HDRBuffer(bool highPrecision = true);
  virtual ~HDRBuffer();

  // Binds DEPTH and one ping-pong target for writing.
  void bind();

  // Subsequent calls will draw to the backbuffer again.
  void unbind();

  // Subsequent calls to bind() will use the ping-pong targets the other way around.
  void doPingPong();

  // Clears all attachements, that is DEPTH to 1.0, and HDR_0 and HDR_1 to vec3(0).
  void clear();

  void  calculateLuminance(ExposureMeteringMode mode);
  float getTotalLuminance() const;
  float getMaximumLuminance() const;

  void          updateGlowMipMap();
  VistaTexture* getGlowMipMap() const;
  VistaTexture* getDepthAttachment() const;
  VistaTexture* getCurrentWriteAttachment() const;
  VistaTexture* getCurrentReadAttachment() const;

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
