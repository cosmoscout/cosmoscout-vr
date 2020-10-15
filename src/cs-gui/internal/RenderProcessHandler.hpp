////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CS_GUI_DETAIL_RENDERPROCESSHANDLER_HPP
#define CS_GUI_DETAIL_RENDERPROCESSHANDLER_HPP

#include <include/cef_render_process_handler.h>

namespace cs::gui::detail {

/// Used to add the callNative method to the JS window object.
class RenderProcessHandler : public CefRenderProcessHandler {
 public:
  /// This is called for each new context. We use this callback to add the
  /// callNative method to the window object.
  void OnContextCreated(CefRefPtr<CefBrowser> browser, CefRefPtr<CefFrame> frame,
      CefRefPtr<CefV8Context> context) override;

 private:
  IMPLEMENT_REFCOUNTING(RenderProcessHandler);
};

} // namespace cs::gui::detail

#endif // CS_GUI_DETAIL_RENDERPROCESSHANDLER_HPP
