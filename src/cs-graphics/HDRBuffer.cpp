////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "HDRBuffer.hpp"

#include "GlowMipMap.hpp"
#include "LuminanceMipMap.hpp"

#include <VistaKernel/DisplayManager/VistaDisplayManager.h>
#include <VistaKernel/DisplayManager/VistaViewport.h>
#include <VistaKernel/VistaSystem.h>
#include <VistaOGLExt/VistaFramebufferObj.h>
#include <VistaOGLExt/VistaGLSLShader.h>

#include <chrono>

namespace cs::graphics {

////////////////////////////////////////////////////////////////////////////////////////////////////

HDRBuffer::HDRBuffer(bool highPrecision)
    : mHighPrecision(highPrecision) {
}

////////////////////////////////////////////////////////////////////////////////////////////////////

HDRBuffer::~HDRBuffer() {
  for (auto& hdrBuffer : mHDRBufferData) {
    delete hdrBuffer.second.mFBO;
    for (auto tex : hdrBuffer.second.mColorAttachments) {
      delete tex;
    }

    delete hdrBuffer.second.mLuminanceMipMap;
  }
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

    // Clear old framebuffer object.
    delete hdrBuffer.mFBO;

    // Create new framebuffer object.
    hdrBuffer.mFBO = new VistaFramebufferObj();

    // Attaches a new texture to the hdrBuffer framebuffer object.
    auto addAttachment = [&hdrBuffer](VistaTexture*& texture, int attachment, int internalFormat,
                             int format, int type) {
      if (!texture) {
        texture = new VistaTexture(GL_TEXTURE_2D);
      }

      texture->Bind();
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
      glTexImage2D(GL_TEXTURE_2D, 0, internalFormat, hdrBuffer.mWidth, hdrBuffer.mHeight, 0, format,
          type, nullptr);
      hdrBuffer.mFBO->Attach(texture, attachment);
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
    delete hdrBuffer.mLuminanceMipMap;
    hdrBuffer.mLuminanceMipMap = new LuminanceMipMap(size[0], size[1]);

    // Create glow mipmaps.
    delete hdrBuffer.mGlowMipMap;
    hdrBuffer.mGlowMipMap = new GlowMipMap(size[0], size[1]);
  }

  // Bind the framebuffer object for writing.
  if (!hdrBuffer.mIsBound) {
    glGetIntegerv(GL_VIEWPORT, hdrBuffer.mCachedViewport);
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
    glViewport(hdrBuffer.mCachedViewport[0], hdrBuffer.mCachedViewport[1],
        hdrBuffer.mCachedViewport[2], hdrBuffer.mCachedViewport[3]);
    glScissor(hdrBuffer.mCachedViewport[0], hdrBuffer.mCachedViewport[1],
        hdrBuffer.mCachedViewport[2], hdrBuffer.mCachedViewport[3]);
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
  glClearColor(0.f, 0.f, 0.f, 0.f);
  glClearDepth(1.0);

  GLenum bufs[] = {GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1};
  glDrawBuffers(2, bufs);

  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

VistaTexture* HDRBuffer::getDepthAttachment() const {
  auto& hdrBuffer = getCurrentHDRBuffer();
  return hdrBuffer.mDepthAttachment;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

VistaTexture* HDRBuffer::getCurrentWriteAttachment() const {
  auto& hdrBuffer = getCurrentHDRBuffer();
  if (hdrBuffer.mCompositePinpongState == 0) {
    return hdrBuffer.mColorAttachments[0];
  }
  return hdrBuffer.mColorAttachments[1];
}

////////////////////////////////////////////////////////////////////////////////////////////////////

VistaTexture* HDRBuffer::getCurrentReadAttachment() const {
  auto& hdrBuffer = getCurrentHDRBuffer();
  if (hdrBuffer.mCompositePinpongState == 0) {
    return hdrBuffer.mColorAttachments[1];
  }
  return hdrBuffer.mColorAttachments[0];
}

////////////////////////////////////////////////////////////////////////////////////////////////////

HDRBuffer::HDRBufferData& HDRBuffer::getCurrentHDRBuffer() {
  auto viewport = GetVistaSystem()->GetDisplayManager()->GetCurrentRenderInfo()->m_pViewport;
  return mHDRBufferData[viewport];
}

////////////////////////////////////////////////////////////////////////////////////////////////////

HDRBuffer::HDRBufferData const& HDRBuffer::getCurrentHDRBuffer() const {
  auto viewport = GetVistaSystem()->GetDisplayManager()->GetCurrentRenderInfo()->m_pViewport;
  return mHDRBufferData.find(viewport)->second;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::array<int, 2> HDRBuffer::getCurrentViewPortSize() {
  auto viewport = GetVistaSystem()->GetDisplayManager()->GetCurrentRenderInfo()->m_pViewport;
  std::array<int, 2> size{};
  viewport->GetViewportProperties()->GetSize(size[0], size[1]);
  return size;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::array<int, 2> HDRBuffer::getCurrentViewPortPos() {
  auto viewport = GetVistaSystem()->GetDisplayManager()->GetCurrentRenderInfo()->m_pViewport;
  std::array<int, 2> pos{};
  viewport->GetViewportProperties()->GetPosition(pos[0], pos[1]);
  return pos;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void HDRBuffer::calculateLuminance() {
  auto&         hdrBuffer = getCurrentHDRBuffer();
  VistaTexture* composite;

  if (hdrBuffer.mCompositePinpongState == 0) {
    composite = hdrBuffer.mColorAttachments[0];
  } else {
    composite = hdrBuffer.mColorAttachments[1];
  }

  hdrBuffer.mLuminanceMipMap->update(composite);

  if (hdrBuffer.mLuminanceMipMap->getLastTotalLuminance()) {
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

void HDRBuffer::updateGlowMipMap() {
  auto&         hdrBuffer = getCurrentHDRBuffer();
  VistaTexture* composite;

  if (hdrBuffer.mCompositePinpongState == 0) {
    composite = hdrBuffer.mColorAttachments[0];
  } else {
    composite = hdrBuffer.mColorAttachments[1];
  }

  hdrBuffer.mGlowMipMap->update(composite);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

VistaTexture* HDRBuffer::getGlowMipMap() const {
  auto& hdrBuffer = getCurrentHDRBuffer();
  return hdrBuffer.mGlowMipMap;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cs::graphics
