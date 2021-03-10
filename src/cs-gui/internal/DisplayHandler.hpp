////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CS_GUI_DETAIL_DISPLAYHANDLER_HPP
#define CS_GUI_DETAIL_DISPLAYHANDLER_HPP

#include "../types.hpp"

#include <include/cef_client.h>

namespace cs::gui::detail {

/// This implementation of CefDisplayHandler is used to forward browser console messages to the
/// standard output.
class DisplayHandler : public CefDisplayHandler {

 public:
  /// The given callback is fired when the cursor icon should change.
  void SetCursorChangeCallback(CursorChangeCallback const& callback);

  /// Prints browser console log messages to the standard output.
  bool OnConsoleMessage(CefRefPtr<CefBrowser> browser, cef_log_severity_t level,
      CefString const& message, CefString const& source, int line) override;

  /// Implements the custom cursor change handling.
  bool OnCursorChange(CefRefPtr<CefBrowser> browser, CefCursorHandle cursor, cef_cursor_type_t type,
      CefCursorInfo const& custom_cursor_info) override;

 private:
  IMPLEMENT_REFCOUNTING(DisplayHandler);

  CursorChangeCallback mCursorChangeCallback;
};

} // namespace cs::gui::detail

#endif // CS_GUI_DETAIL_DISPLAYHANDLER_HPP
