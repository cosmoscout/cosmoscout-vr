////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "GuiItem.hpp"

#include "GuiArea.hpp"

#include <VistaOGLExt/VistaTexture.h>

namespace cs::gui {

////////////////////////////////////////////////////////////////////////////////////////////////////

GuiItem::GuiItem(std::string const& url, bool allowLocalFileAccess)
    : WebView(url, 100, 100, allowLocalFileAccess)
    , mAreaWidth(1)
    , mAreaHeight(1)
    , mSizeX(0)
    , mSizeY(0)
    , mPositionX(0)
    , mPositionY(0)
    , mOffsetX(0)
    , mOffsetY(0)
    , mRelSizeX(1.F)
    , mRelSizeY(1.F)
    , mRelPositionX(0.5F)
    , mRelPositionY(0.5F)
    , mRelOffsetX(0.F)
    , mRelOffsetY(0.F)
    , mIsRelSizeX(true)
    , mIsRelSizeY(true)
    , mIsRelPositionX(true)
    , mIsRelPositionY(true)
    , mIsRelOffsetX(true)
    , mIsRelOffsetY(true) {

  setDrawCallback([this](DrawEvent const& event) {
    mPBOsNeedUpload = static_cast<uint8_t>(mTexturePBOs.size());

    if (event.mResized) {
      mTextureSizeX = event.mWidth;
      mTextureSizeY = event.mHeight;

      recreateBuffers();
    }

    return mPixels.data();
  });

  setRequestKeyboardFocusCallback(
      [this](bool requested) { mIsKeyboardInputElementFocused = requested; });

  mTextureSizeX = getWidth();
  mTextureSizeY = getHeight();

  recreateBuffers();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

GuiItem::~GuiItem() {
  glDeleteBuffers(2, &mTexturePBOs[0]);
  glDeleteTextures(1, &mTexture);

  // seems to be necessary as OnPaint can be called by some other thread even
  // if this object is already deleted
  setDrawCallback([](DrawEvent const& /*unused*/) { return nullptr; });
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void GuiItem::setSizeX(unsigned int value) {
  mSizeX      = value;
  mIsRelSizeX = false;
  updateSizes();
}
void GuiItem::setSizeY(unsigned int value) {
  mSizeY      = value;
  mIsRelSizeY = false;
  updateSizes();
}
void GuiItem::setPositionX(int value) {
  mPositionX      = value;
  mIsRelPositionX = false;
  updateSizes();
}
void GuiItem::setPositionY(int value) {
  mPositionY      = value;
  mIsRelPositionY = false;
  updateSizes();
}
void GuiItem::setOffsetX(int value) {
  mOffsetX      = value;
  mIsRelOffsetX = false;
  updateSizes();
}
void GuiItem::setOffsetY(int value) {
  mOffsetY      = value;
  mIsRelOffsetY = false;
  updateSizes();
}
void GuiItem::setRelSizeX(float value) {
  mRelSizeX   = value;
  mIsRelSizeX = true;
  updateSizes();
}
void GuiItem::setRelSizeY(float value) {
  mRelSizeY   = value;
  mIsRelSizeY = true;
  updateSizes();
}
void GuiItem::setRelPositionX(float value) {
  mRelPositionX   = value;
  mIsRelPositionX = true;
  updateSizes();
}
void GuiItem::setRelPositionY(float value) {
  mRelPositionY   = value;
  mIsRelPositionY = true;
  updateSizes();
}
void GuiItem::setRelOffsetX(float value) {
  mRelOffsetX   = value;
  mIsRelOffsetX = true;
  updateSizes();
}
void GuiItem::setRelOffsetY(float value) {
  mRelOffsetY   = value;
  mIsRelOffsetY = true;
  updateSizes();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

unsigned int GuiItem::getSizeX() const {
  return mSizeX;
}
unsigned int GuiItem::getSizeY() const {
  return mSizeY;
}
int GuiItem::getPositionX() const {
  return mPositionX;
}
int GuiItem::getPositionY() const {
  return mPositionY;
}
int GuiItem::getOffsetX() const {
  return mOffsetX;
}
int GuiItem::getOffsetY() const {
  return mOffsetY;
}
float GuiItem::getRelSizeX() const {
  return mRelSizeX;
}
float GuiItem::getRelSizeY() const {
  return mRelSizeY;
}
float GuiItem::getRelPositionX() const {
  return mRelPositionX;
}
float GuiItem::getRelPositionY() const {
  return mRelPositionY;
}
float GuiItem::getRelOffsetX() const {
  return mRelOffsetX;
}
float GuiItem::getRelOffsetY() const {
  return mRelOffsetY;
}
int GuiItem::getTextureSizeX() const {
  return mTextureSizeX;
}
int GuiItem::getTextureSizeY() const {
  return mTextureSizeY;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void GuiItem::setIsEnabled(bool bEnabled) {
  mIsEnabled = bEnabled;
}

bool GuiItem::getIsEnabled() const {
  return mIsEnabled;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool GuiItem::getIsKeyboardInputElementFocused() const {
  return mIsKeyboardInputElementFocused;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool GuiItem::calculateMousePosition(int areaX, int areaY, int& x, int& y) {
  int tmpX = areaX - mOffsetX - mPositionX + getWidth() / 2;
  int tmpY = areaY - mOffsetY - mPositionY + getHeight() / 2;

  x = tmpX;
  y = tmpY;

  if (tmpX > static_cast<int>(mSizeX) - 1 || tmpX < 0) {
    return false;
  }

  if (tmpY > static_cast<int>(mSizeY) - 1 || tmpY < 0) {
    return false;
  }

  return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void GuiItem::onAreaResize(int width, int height) {
  mAreaWidth  = width;
  mAreaHeight = height;

  updateSizes();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

uint32_t GuiItem::getTexture() const {

  // Copy the pixels buffer to the texture.
  if (mPBOsNeedUpload > 0) {

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, mTexturePBOs[mCurrentPBO]);
    auto ptr = static_cast<uint8_t*>(glMapBufferRange(
        GL_PIXEL_UNPACK_BUFFER, 0, 4 * mTextureSizeX * mTextureSizeY, GL_MAP_WRITE_BIT));
    std::memcpy(ptr, mPixels.data(), 4 * mTextureSizeX * mTextureSizeY);
    glUnmapBuffer(GL_PIXEL_UNPACK_BUFFER);

    glBindTexture(GL_TEXTURE_2D, mTexture);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER,
        mTexturePBOs[(mCurrentPBO + mTexturePBOs.size() - 1) % mTexturePBOs.size()]);
    glTexSubImage2D(
        GL_TEXTURE_2D, 0, 0, 0, mTextureSizeX, mTextureSizeY, GL_BGRA, GL_UNSIGNED_BYTE, 0);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    glBindTexture(GL_TEXTURE_2D, 0);

    mCurrentPBO = (mCurrentPBO + 1) % mTexturePBOs.size();
    mPBOsNeedUpload -= 1;
  }

  return mTexture;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void GuiItem::recreateBuffers() {

  if (mTexture) {
    glDeleteTextures(1, &mTexture);
    glDeleteBuffers(2, &mTexturePBOs[0]);
  }

  for (size_t i = 0; i < mTexturePBOs.size(); ++i) {
    glGenBuffers(1, &mTexturePBOs[i]);
    glBindBuffer(GL_TEXTURE_BUFFER, mTexturePBOs[i]);
    glBufferStorage(
        GL_TEXTURE_BUFFER, 4 * mTextureSizeX * mTextureSizeY, nullptr, GL_MAP_WRITE_BIT);
  }
  glBindBuffer(GL_TEXTURE_BUFFER, 0);

  glGenTextures(1, &mTexture);
  glBindTexture(GL_TEXTURE_2D, mTexture);
  glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, mTextureSizeX, mTextureSizeY);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glBindTexture(GL_TEXTURE_2D, 0);

  mPixels.resize(4 * mTextureSizeX * mTextureSizeY);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void GuiItem::updateSizes() {
  if (mIsRelSizeX) {
    mSizeX = static_cast<uint32_t>(mRelSizeX * static_cast<float>(mAreaWidth));
  } else {
    mRelSizeX = static_cast<float>(1.0 * mSizeX / mAreaWidth);
  }

  if (mIsRelSizeY) {
    mSizeY = static_cast<uint32_t>(mRelSizeY * static_cast<float>(mAreaHeight));
  } else {
    mRelSizeY = static_cast<float>(1.0 * mSizeY / mAreaHeight);
  }

  if (mIsRelPositionX) {
    mPositionX = static_cast<uint32_t>(mRelPositionX * static_cast<float>(mAreaWidth));
  } else {
    mRelPositionX = static_cast<float>(1.0 * mPositionX / mAreaWidth);
  }

  if (mIsRelPositionY) {
    mPositionY = static_cast<uint32_t>(mRelPositionY * static_cast<float>(mAreaHeight));
  } else {
    mRelPositionY = static_cast<float>(1.0 * mPositionY / mAreaHeight);
  }

  if (mIsRelOffsetX) {
    mOffsetX = static_cast<uint32_t>(mRelOffsetX * static_cast<float>(mAreaWidth));
  } else {
    mRelOffsetX = static_cast<float>(1.0 * mOffsetX / mAreaWidth);
  }

  if (mIsRelOffsetY) {
    mOffsetY = static_cast<uint32_t>(mRelOffsetY * static_cast<float>(mAreaHeight));
  } else {
    mRelOffsetY = static_cast<float>(1.0 * mOffsetY / mAreaHeight);
  }

  resize(static_cast<int>(mSizeX), static_cast<int>(mSizeY));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cs::gui
