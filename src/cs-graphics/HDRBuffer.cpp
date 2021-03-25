////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "HDRBuffer.hpp"

#include "GlareMipMap.hpp"
#include "LuminanceMipMap.hpp"
#include "logger.hpp"

#include <VistaKernel/DisplayManager/VistaDisplayManager.h>
#include <VistaKernel/DisplayManager/VistaViewport.h>
#include <VistaKernel/VistaSystem.h>
#include <VistaOGLExt/VistaFramebufferObj.h>
#include <VistaOGLExt/VistaGLSLShader.h>

#include <chrono>

namespace cs::graphics {

////////////////////////////////////////////////////////////////////////////////////////////////////

HDRBuffer::HDRBuffer(uint32_t multiSamples, bool highPrecision)
    : mMultiSamples(multiSamples)
    , mHighPrecision(highPrecision) {
}

////////////////////////////////////////////////////////////////////////////////////////////////////

HDRBuffer::~HDRBuffer() {
  // Destructor must not be inline as the std::unique_ptrs would otherwise not accept incomplete
  // types in the header.
}

////////////////////////////////////////////////////////////////////////////////////////////////////

uint32_t HDRBuffer::getMultiSamples() const {
  return mMultiSamples;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void HDRBuffer::bind() {

  // There is a different framebuffer object for each viewport. Here wee retrieve the current one.
  auto& hdrBuffer = getCurrentHDRBuffer();
  auto  size      = getCurrentViewPortSize();

  // If the size changed, we have to re-create the framebuffer object and its attachments.
  if (size[0] != hdrBuffer.mWidth || size[1] != hdrBuffer.mHeight) {
    hdrBuffer.mWidth  = size[0];
    hdrBuffer.mHeight = size[1];

    // Create new framebuffer object.
    hdrBuffer.mFBO.reset(new VistaFramebufferObj());

    // Attaches a new texture to the hdrBuffer framebuffer object.
    auto addAttachment = [&hdrBuffer, this](std::unique_ptr<VistaTexture>& texture, int attachment,
                             int internalFormat, int format, int type) {
      auto target = (mMultiSamples > 0) ? GL_TEXTURE_2D_MULTISAMPLE : GL_TEXTURE_2D;

      if (!texture) {
        texture = std::make_unique<VistaTexture>(target);
      }

      texture->Bind();

      if (mMultiSamples > 0) {
        glTexImage2DMultisample(
            target, mMultiSamples, internalFormat, hdrBuffer.mWidth, hdrBuffer.mHeight, false);
      } else {
        glTexParameteri(target, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(target, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(target, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(target, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexImage2D(target, 0, internalFormat, hdrBuffer.mWidth, hdrBuffer.mHeight, 0, format,
            type, nullptr);
      }
      hdrBuffer.mFBO->Attach(texture.get(), attachment);
    };

    // Add HDR-attachment ping-pong A.
    addAttachment(hdrBuffer.mColorAttachments[0], GL_COLOR_ATTACHMENT0,
        mHighPrecision ? GL_RGBA32F : GL_RGBA16F, GL_RGBA, GL_FLOAT);

    // Add HDR-attachment ping-pong B.
    addAttachment(hdrBuffer.mColorAttachments[1], GL_COLOR_ATTACHMENT1,
        mHighPrecision ? GL_RGBA32F : GL_RGBA16F, GL_RGBA, GL_FLOAT);

    // Add depth-attachment.
    addAttachment(hdrBuffer.mDepthAttachment, GL_DEPTH_ATTACHMENT, GL_DEPTH_COMPONENT24,
        GL_DEPTH_COMPONENT, GL_FLOAT);

    // Create luminance mipmaps.
    hdrBuffer.mLuminanceMipMap.reset(new LuminanceMipMap(mMultiSamples, size[0], size[1]));

    // Create glare mipmaps.
    hdrBuffer.mGlareMipMap.reset(new GlareMipMap(mMultiSamples, size[0], size[1]));
  }

  // Bind the framebuffer object for writing.
  if (!hdrBuffer.mIsBound) {
    glGetIntegerv(GL_VIEWPORT, hdrBuffer.mCachedViewport.data());
    hdrBuffer.mFBO->Bind();
    hdrBuffer.mIsBound = true;
  }

  // Select the write attachment based on the current ping-pong state.
  if (hdrBuffer.mCompositePinpongState == 0) {
    glDrawBuffer(GL_COLOR_ATTACHMENT0);
  } else {
    glDrawBuffer(GL_COLOR_ATTACHMENT1);
  }

  // Set the viewport to the entire hdrBuffer.
  glViewport(0, 0, hdrBuffer.mWidth, hdrBuffer.mHeight);
  glScissor(0, 0, hdrBuffer.mWidth, hdrBuffer.mHeight);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void HDRBuffer::unbind() {

  // There is a different framebuffer object for each viewport. Here wee retrieve the current one.
  auto& hdrBuffer = getCurrentHDRBuffer();

  // If it is bound, unbind it.
  if (hdrBuffer.mIsBound) {
    hdrBuffer.mFBO->Release();
    hdrBuffer.mIsBound = false;

    // And restore the original viewport.
    glViewport(hdrBuffer.mCachedViewport.at(0), hdrBuffer.mCachedViewport.at(1),
        hdrBuffer.mCachedViewport.at(2), hdrBuffer.mCachedViewport.at(3));
    glScissor(hdrBuffer.mCachedViewport.at(0), hdrBuffer.mCachedViewport.at(1),
        hdrBuffer.mCachedViewport.at(2), hdrBuffer.mCachedViewport.at(3));
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void HDRBuffer::doPingPong() {
  // There is a different framebuffer object for each viewport. Here wee retrieve the current one.
  auto& hdrBuffer                  = getCurrentHDRBuffer();
  hdrBuffer.mCompositePinpongState = (hdrBuffer.mCompositePinpongState + 1) % 2;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void HDRBuffer::clear() {
  bind();
  glClearColor(0.F, 0.F, 0.F, 0.F);
  glClearDepth(1.0);

  std::array<GLenum, 2> bufs = {GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1};
  glDrawBuffers(2, bufs.data());

  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

VistaTexture* HDRBuffer::getDepthAttachment() const {
  auto const& hdrBuffer = getCurrentHDRBuffer();
  return hdrBuffer.mDepthAttachment.get();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

VistaTexture* HDRBuffer::getCurrentWriteAttachment() const {
  auto const& hdrBuffer = getCurrentHDRBuffer();
  if (hdrBuffer.mCompositePinpongState == 0) {
    return hdrBuffer.mColorAttachments.at(0).get();
  }
  return hdrBuffer.mColorAttachments.at(1).get();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

VistaTexture* HDRBuffer::getCurrentReadAttachment() const {
  auto const& hdrBuffer = getCurrentHDRBuffer();
  if (hdrBuffer.mCompositePinpongState == 0) {
    return hdrBuffer.mColorAttachments.at(1).get();
  }
  return hdrBuffer.mColorAttachments.at(0).get();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

HDRBuffer::HDRBufferData& HDRBuffer::getCurrentHDRBuffer() {
  auto* viewport = GetVistaSystem()->GetDisplayManager()->GetCurrentRenderInfo()->m_pViewport;
  return mHDRBufferData[viewport];
}

////////////////////////////////////////////////////////////////////////////////////////////////////

HDRBuffer::HDRBufferData const& HDRBuffer::getCurrentHDRBuffer() const {
  auto* viewport = GetVistaSystem()->GetDisplayManager()->GetCurrentRenderInfo()->m_pViewport;
  return mHDRBufferData.find(viewport)->second;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::array<int, 2> HDRBuffer::getCurrentViewPortSize() {
  auto* viewport = GetVistaSystem()->GetDisplayManager()->GetCurrentRenderInfo()->m_pViewport;
  std::array<int, 2> size{};
  viewport->GetViewportProperties()->GetSize(size.at(0), size.at(1));
  return size;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::array<int, 2> HDRBuffer::getCurrentViewPortPos() {
  auto* viewport = GetVistaSystem()->GetDisplayManager()->GetCurrentRenderInfo()->m_pViewport;
  std::array<int, 2> pos{};
  viewport->GetViewportProperties()->GetPosition(pos.at(0), pos.at(1));
  return pos;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void HDRBuffer::calculateLuminance() {
  auto&         hdrBuffer = getCurrentHDRBuffer();
  VistaTexture* composite = nullptr;

  if (hdrBuffer.mCompositePinpongState == 0) {
    composite = hdrBuffer.mColorAttachments.at(0).get();
  } else {
    composite = hdrBuffer.mColorAttachments.at(1).get();
  }

  hdrBuffer.mLuminanceMipMap->update(composite);

  if (hdrBuffer.mLuminanceMipMap->getLastTotalLuminance() != 0.0F) {
    mTotalLuminance   = hdrBuffer.mLuminanceMipMap->getLastTotalLuminance();
    mMaximumLuminance = hdrBuffer.mLuminanceMipMap->getLastMaximumLuminance();
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

float HDRBuffer::getTotalLuminance() const {
  return mTotalLuminance;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

float HDRBuffer::getMaximumLuminance() const {
  return mMaximumLuminance;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void HDRBuffer::updateGlareMipMap() {
  auto&         hdrBuffer = getCurrentHDRBuffer();
  VistaTexture* composite = nullptr;

  if (hdrBuffer.mCompositePinpongState == 0) {
    composite = hdrBuffer.mColorAttachments.at(0).get();
  } else {
    composite = hdrBuffer.mColorAttachments.at(1).get();
  }

  hdrBuffer.mGlareMipMap->update(composite, mGlareMode, mGlareQuality);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

VistaTexture* HDRBuffer::getGlareMipMap() const {
  auto const& hdrBuffer = getCurrentHDRBuffer();
  return hdrBuffer.mGlareMipMap.get();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void HDRBuffer::setGlareMode(HDRBuffer::GlareMode value) {
  mGlareMode = value;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

HDRBuffer::GlareMode HDRBuffer::getGlareMode() const {
  return mGlareMode;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void HDRBuffer::setGlareQuality(uint32_t quality) {
  mGlareQuality = quality;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

uint32_t HDRBuffer::getGlareQuality() const {
  return mGlareQuality;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cs::graphics
