////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "GuiItem.hpp"

#include "GuiArea.hpp"

#include <VistaOGLExt/VistaTexture.h>

namespace cs::gui {

////////////////////////////////////////////////////////////////////////////////////////////////////

GuiItem::GuiItem(std::string const& url, bool allowLocalFileAccess)
    : WebView(url, 100, 100, allowLocalFileAccess)
    //, mTexture(new VistaTexture(GL_TEXTURE_2D))
    , mAreaWidth(1)
    , mAreaHeight(1)
    , mSizeX(0)
    , mSizeY(0)
    , mPositionX(0)
    , mPositionY(0)
    , mOffsetX(0)
    , mOffsetY(0)
    , mRelSizeX(1.f)
    , mRelSizeY(1.f)
    , mRelPositionX(0.5f)
    , mRelPositionY(0.5f)
    , mRelOffsetX(0.f)
    , mRelOffsetY(0.f)
    , mIsRelSizeX(true)
    , mIsRelSizeY(true)
    , mIsRelPositionX(true)
    , mIsRelPositionY(true)
    , mIsRelOffsetX(true)
    , mIsRelOffsetY(true) {
  setDrawCallback([this](DrawEvent const& event) { updateTexture(event); });


  glGenBuffers(1, &mTextureBuffer);
  glBindBuffer(GL_TEXTURE_BUFFER, mTextureBuffer);
  glBufferData(GL_TEXTURE_BUFFER, 4 * sizeof(uint8_t) * getWidth() * getHeight(), nullptr, GL_STATIC_DRAW);

  glGenTextures(1, &mTexture);
  glBindBuffer(GL_TEXTURE_BUFFER, 0);

  //mTexture->SetWrapS(GL_CLAMP_TO_EDGE);
  //mTexture->SetWrapT(GL_CLAMP_TO_EDGE);
  //mTexture->UploadTexture(getWidth(), getHeight(), nullptr, false);
  //mTexture->Unbind();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

GuiItem::~GuiItem() {
  // seems to be necessary as OnPaint can be called by some other thread even
  // if this object is already deleted
  setDrawCallback([](DrawEvent const& event) {});
  //delete mTexture;
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

////////////////////////////////////////////////////////////////////////////////////////////////////

void GuiItem::setIsEnabled(bool bEnabled) {
  mIsEnabled = bEnabled;
}

bool GuiItem::getIsEnabled() const {
  return mIsEnabled;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool GuiItem::calculateMousePosition(int areaX, int areaY, int& x, int& y) {
  int tmpX = areaX - mOffsetX - mPositionX + getWidth() / 2;
  int tmpY = areaY - mOffsetY - mPositionY + getHeight() / 2;

  x = tmpX;
  y = tmpY;

  if (tmpX > mSizeX - 1 || tmpX < 0)
    return false;
  if (tmpY > mSizeY - 1 || tmpY < 0)
    return false;

  return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void GuiItem::updateTexture(DrawEvent const& event) {
  glBindBuffer(GL_TEXTURE_BUFFER, mTextureBuffer);

  if (event.mResized) {
    glBufferData(GL_TEXTURE_BUFFER, 4 * sizeof(uint8_t) * event.mWidth * event.mHeight, event.mData, GL_STATIC_DRAW);
  } else {
    glBufferSubData(GL_TEXTURE_BUFFER, 0, 4 * sizeof(uint8_t) * event.mWidth * event.mHeight, event.mData);
  }


  glBindBuffer(GL_TEXTURE_BUFFER, 0);

  //mTexture->Bind();

  /*
  if (event.mResized) {
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, event.mWidth, event.mHeight, 0, GL_BGRA,
        GL_UNSIGNED_BYTE, event.mData);
  } else {
    glTexSubImage2D(GL_TEXTURE_2D, 0, event.mX, event.mY, event.mWidth, event.mHeight, GL_BGRA,
        GL_UNSIGNED_BYTE, event.mData);
  }*/

  //mTexture->Unbind();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void GuiItem::onAreaResize(int width, int height) {
  mAreaWidth  = width;
  mAreaHeight = height;

  updateSizes();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void GuiItem::updateSizes() {
  if (mIsRelSizeX) {
    mSizeX = mRelSizeX * mAreaWidth;
  } else {
    mRelSizeX = 1.f * mSizeX / mAreaWidth;
  }

  if (mIsRelSizeY) {
    mSizeY = mRelSizeY * mAreaHeight;
  } else {
    mRelSizeY = 1.f * mSizeY / mAreaHeight;
  }

  if (mIsRelPositionX) {
    mPositionX = mRelPositionX * mAreaWidth;
  } else {
    mRelPositionX = 1.f * mPositionX / mAreaWidth;
  }

  if (mIsRelPositionY) {
    mPositionY = mRelPositionY * mAreaHeight;
  } else {
    mRelPositionY = 1.f * mPositionY / mAreaHeight;
  }

  if (mIsRelOffsetX) {
    mOffsetX = mRelOffsetX * mAreaWidth;
  } else {
    mRelOffsetX = 1.f * mOffsetX / mAreaWidth;
  }

  if (mIsRelOffsetY) {
    mOffsetY = mRelOffsetY * mAreaHeight;
  } else {
    mRelOffsetY = 1.f * mOffsetY / mAreaHeight;
  }

  resize(mSizeX, mSizeY);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::pair<uint32_t, uint32_t> GuiItem::getTexture() const {
  return {mTextureBuffer, mTexture};
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cs::gui
