////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CS_GUI_DETAIL_DISPLAYHANDLER_HPP
#define CS_GUI_DETAIL_DISPLAYHANDLER_HPP

#include <include/cef_client.h>

namespace cs::gui::detail {

/// This implementation of CefDisplayHandler is used to forward browser console messages to the
/// standard output.
class DisplayHandler : public CefDisplayHandler {

 public:
  /// Prints browser console log messages to the standard output.
  bool OnConsoleMessage(CefRefPtr<CefBrowser> browser, CefString const& message,
      CefString const& source, int line) override;

 private:
  IMPLEMENT_REFCOUNTING(DisplayHandler);
};

} // namespace cs::gui::detail

#endif // CS_GUI_DETAIL_DISPLAYHANDLER_HPP
