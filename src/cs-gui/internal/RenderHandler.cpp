////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "RenderHandler.hpp"
#include "../../cs-utils/FrameTimings.hpp"
#include "../logger.hpp"

#include <GL/glew.h>
#include <deque>
#include <thread>

namespace cs::gui::detail {

////////////////////////////////////////////////////////////////////////////////////////////////////

void RenderHandler::SetDrawCallback(DrawCallback const& callback) {
  mDrawCallback = callback;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void RenderHandler::SetRequestKeyboardFocusCallback(RequestKeyboardFocusCallback const& callback) {
  mRequestKeyboardFocusCallback = callback;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void RenderHandler::Resize(int width, int height) {
  mWidth  = width;
  mHeight = height;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool RenderHandler::GetColor(int x, int y, uint8_t& r, uint8_t& g, uint8_t& b, uint8_t& a) const {
  if (!mPixelData) {
    return false;
  }

  int data_pos(x * 4 + y * mLastDrawWidth * 4);

  if (data_pos + 3 >= 4 * mWidth * mHeight) {
    return false;
  }

  // this might be dangerous --- I'm not entirely sure whether this pixel data
  // reference is guranteed to be valid. If something bad happens, we have to
  // consider keeping a local copy of the pixel data...
  b = mPixelData[data_pos + 0]; // NOLINT: This
  g = mPixelData[data_pos + 1]; // NOLINT: is
  r = mPixelData[data_pos + 2]; // NOLINT: performance
  a = mPixelData[data_pos + 3]; // NOLINT: critical.

  return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

int RenderHandler::GetWidth() const {
  return mWidth;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

int RenderHandler::GetHeight() const {
  return mHeight;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void RenderHandler::GetViewRect(CefRefPtr<CefBrowser> /*browser*/, CefRect& rect) {
  rect = CefRect(0, 0, mWidth, mHeight);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void RenderHandler::OnPaint(CefRefPtr<CefBrowser> /*browser*/, PaintElementType /*type*/,
    RectList const& dirtyRects, const void* b, int width, int height) {
  DrawEvent event{};
  event.mResized  = width != mLastDrawWidth || height != mLastDrawHeight;
  mLastDrawWidth  = width;
  mLastDrawHeight = height;

  if (event.mResized) {
    event.mX      = 0;
    event.mY      = 0;
    event.mWidth  = width;
    event.mHeight = height;
  }

  mPixelData = mDrawCallback(event);
  if (!mPixelData) {
    std::cerr << "[" << __FILE__ << ":" << __LINE__
              << "] Error when initializing GUI Texture Buffer!" << std::endl;
    return;
  }

  if (event.mResized) {
    size_t bufferSize = width * height * 4;
    std::memcpy(mPixelData, b, bufferSize * sizeof(uint8_t));
  } else {
    for (auto const& rect : dirtyRects) {
      if (rect.width > 0.5 * width) {
        // When the rect is almost the whole screen width we just copy the rest of the width
        // too. This is faster since we only need one efficient std::memcpy call.

        size_t startOffset = rect.y * width * 4 * sizeof(uint8_t);
        size_t extend      = rect.height * width * 4 * sizeof(uint8_t);

        // NOLINTNEXTLINE: This is performance critical.
        std::memcpy(mPixelData + startOffset, (uint8_t*)b + startOffset, extend);
      } else {
        // We copy each row of the changed region over individually, since they are not
        // guaranteed to have continuous memory.
        //
        // ################################################################################
        // ##############################+--------------------------------------+##########
        // ####################### i = 0 |~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|##########
        // ####################### i = 1 |~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|##########
        // ####################### i = 2 |~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|##########
        // ####################### i = 3 |~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|##########
        // ####################### i = 4 |~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|##########
        // ####################### i = 5 |~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|##########
        // ##############################+--------------------------------------+##########
        // ################################################################################
        // ################################################################################
        for (int i = 0; i < rect.height; ++i) {
          size_t startOffset = ((rect.y + i) * width + rect.x) * 4 * sizeof(uint8_t);
          size_t extend      = rect.width * 4 * sizeof(uint8_t);

          // NOLINTNEXTLINE: This is performance critical.
          std::memcpy(mPixelData + startOffset, (uint8_t*)b + startOffset, extend);
        }
      }
    }
  }
} // namespace cs::gui::detail

////////////////////////////////////////////////////////////////////////////////////////////////////

void RenderHandler::OnVirtualKeyboardRequested(
    CefRefPtr<CefBrowser> /*browser*/, TextInputMode input_mode) {

  if (mRequestKeyboardFocusCallback) {
    mRequestKeyboardFocusCallback(input_mode != CEF_TEXT_INPUT_MODE_NONE);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cs::gui::detail
