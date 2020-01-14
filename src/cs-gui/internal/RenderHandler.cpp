////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "RenderHandler.hpp"
#include "../../cs-utils/FrameTimings.hpp"
#include <chrono>
#include <deque>
#include <iostream>

namespace cs::gui::detail {

////////////////////////////////////////////////////////////////////////////////////////////////////

void RenderHandler::SetDrawCallback(DrawCallback const& callback) {
  mDrawCallback = callback;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void RenderHandler::SetCursorChangeCallback(CursorChangeCallback const& callback) {
  mCursorChangeCallback = callback;
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

  // this might be dangerous --- I'm not entirely sure whether this pixel data
  // reference is guranteed to be valid. If something bad happens, we have to
  // consider keeping a local copy of the pixel data...
  b = mPixelData[data_pos + 0];
  g = mPixelData[data_pos + 1];
  r = mPixelData[data_pos + 2];
  a = mPixelData[data_pos + 3];

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

void RenderHandler::GetViewRect(CefRefPtr<CefBrowser> browser, CefRect& rect) {
  rect = CefRect(0, 0, mWidth, mHeight);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

int counter = 0;

void RenderHandler::OnPaint(CefRefPtr<CefBrowser> browser, PaintElementType type,
    RectList const& dirtyRects, const void* b, int width, int height) {
  auto timer = cs::utils::FrameTimings::ScopedTimer("Copy stuff");

  if (++counter == 60) {
    counter = 0;
  }

  auto startCopy1 = std::chrono::high_resolution_clock::now();

  size_t bufferSize = width * height * 4;

  // When the source buffer got larger we reallocate and copy the whole source buffer over.
  if (mCurrentBufferSize < bufferSize) {
    delete[] mPixelData;

    mPixelData         = new uint8_t[bufferSize];
    mCurrentBufferSize = bufferSize;
    std::memcpy(mPixelData, b, bufferSize * sizeof(uint8_t));

    // Otherwise we only copy the dirty regions.
  } else {
    // For each changed region
    for (const auto& rect : dirtyRects) {

      // We copy each row of the changed region over individually, since they are not guaranteed to
      // have continuous memory.
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
        std::memcpy(mPixelData + startOffset, (uint8_t*)b + startOffset, extend);
      }
    }
  }

  if (counter == 0) {
    auto endCopy1 = std::chrono::high_resolution_clock::now();
    auto elapsed1 = std::chrono::duration_cast<std::chrono::microseconds>(endCopy1 - startCopy1).count();
    std::cout << "  Copy1: " << elapsed1 << std::endl;
  }

  if (mDrawCallback) {
    DrawEvent event;
    event.mResized = width != mLastDrawWidth || height != mLastDrawHeight;

    mLastDrawWidth  = width;
    mLastDrawHeight = height;

    //if (event.mResized) {
      auto startCopy2 = std::chrono::high_resolution_clock::now();

      event.mX      = 0;
      event.mY      = 0;
      event.mWidth  = width;
      event.mHeight = height;
      event.mData   = mPixelData;

      mDrawCallback(event);

      if (counter == 0) {
        auto endCopy2 = std::chrono::high_resolution_clock::now();
        auto elapsed2 = std::chrono::duration_cast<std::chrono::microseconds>(endCopy2 - startCopy2).count();
        std::cout << "  Copy2: " << elapsed2 << std::endl;
      }
    /*} else {
      auto startCopy3 = std::chrono::high_resolution_clock::now();

      for (auto const& rect : dirtyRects) {
        event.mX      = rect.x;
        event.mY      = rect.y;
        event.mWidth  = rect.width;
        event.mHeight = rect.height;

        std::vector<uint8_t> data(rect.width * rect.height * 4ul);

        for (int y(0); y < rect.height; ++y) {
          std::memcpy(&data[y * rect.width * 4], mPixelData + ((y + rect.y) * width + rect.x) * 4,
              rect.width * (size_t)4);
        }

        event.mData = data.data();
        mDrawCallback(event);

        if (counter == 0) {
          auto endCopy3 = std::chrono::high_resolution_clock::now();
          auto elapsed3 = std::chrono::duration_cast<std::chrono::microseconds>(endCopy3 - startCopy3).count();
          std::cout << "  Copy3: " << elapsed3 << std::endl;
        }
      }
    }*/
  }

  if (counter == 0) {
    auto endCopyAll = std::chrono::high_resolution_clock::now();
    auto elapsedAll = std::chrono::duration_cast<std::chrono::microseconds>(endCopyAll - startCopy1).count();
    std::cout << "CopyAll: " << elapsedAll << std::endl;
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void RenderHandler::OnCursorChange(CefRefPtr<CefBrowser> browser, CefCursorHandle cursor,
    CefRenderHandler::CursorType type, const CefCursorInfo& customursor_info) {
  if (mCursorChangeCallback) {
    mCursorChangeCallback(static_cast<Cursor>(type));
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

RenderHandler::~RenderHandler() {
  delete[] mPixelData;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cs::gui::detail
