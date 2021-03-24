////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CS_GUI_DETAIL_RENDERHANDLER_HPP
#define CS_GUI_DETAIL_RENDERHANDLER_HPP

#include "../types.hpp"

#include <include/cef_client.h>
#include <include/cef_render_handler.h>

namespace cs::gui::detail {

/// Implements the custom render pipeline, since we are not rendering to a browser window we have to
/// handle the changes of the displayed data on our own.
class RenderHandler : public CefRenderHandler {

 public:
  /// Sets what should happen, when the displayed data changes.
  void SetDrawCallback(DrawCallback const& callback);

  /// The given callback is fired when the active gui element wants to receive keyboard events.
  void SetRequestKeyboardFocusCallback(RequestKeyboardFocusCallback const& callback);

  void Resize(int width, int height);
  bool GetColor(int x, int y, uint8_t& r, uint8_t& g, uint8_t& b, uint8_t& a) const;

  int GetWidth() const;
  int GetHeight() const;

  /// Gives the browser the available area for its view.
  void GetViewRect(CefRefPtr<CefBrowser> browser, CefRect& rect) override;

  /// Implements the custom rendering pipeline.
  void OnPaint(CefRefPtr<CefBrowser> browser, PaintElementType type, const RectList& dirtyRects,
      const void* buffer, int width, int height) override;

  void OnVirtualKeyboardRequested(CefRefPtr<CefBrowser> browser, TextInputMode input_mode) override;

 private:
  IMPLEMENT_REFCOUNTING(RenderHandler);

  int mWidth{};
  int mHeight{};

  int mLastDrawWidth{};
  int mLastDrawHeight{};

  DrawCallback                 mDrawCallback;
  RequestKeyboardFocusCallback mRequestKeyboardFocusCallback;

  uint8_t* mPixelData = nullptr;
};

} // namespace cs::gui::detail

#endif // CS_GUI_DETAIL_RENDERHANDLER_HPP
