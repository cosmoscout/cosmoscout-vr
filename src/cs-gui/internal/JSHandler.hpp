////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CS_GUI_DETAIL_JS_HANDLER_HPP
#define CS_GUI_DETAIL_JS_HANDLER_HPP

#include <include/cef_v8.h>

namespace cs::gui::detail {

/// Handles function calls from Javascript. Only functions with the name "callNative" and at
/// least one argument are accepted.
class JSHandler : public CefV8Handler {

 public:
  explicit JSHandler(CefRefPtr<CefBrowser> const& browser)
      : mBrowser(browser) {
  }

  /// Handles function calls from Javascript. Only functions with the name "callNative" and at
  /// least one argument are accepted.
  bool Execute(const CefString& name, CefRefPtr<CefV8Value> object, const CefV8ValueList& arguments,
      CefRefPtr<CefV8Value>& retval, CefString& exception) override;

 private:
  IMPLEMENT_REFCOUNTING(JSHandler);

  void SendError(std::string const& message) const;

  CefRefPtr<CefBrowser> mBrowser;
};

} // namespace cs::gui::detail

#endif // CS_GUI_DETAIL_JS_HANDLER_HPP
