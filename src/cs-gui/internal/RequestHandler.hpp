////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CS_GUI_DETAIL_REQUESTHANDLER_HPP
#define CS_GUI_DETAIL_REQUESTHANDLER_HPP

#include "ResourceRequestHandler.hpp"

#include <include/cef_client.h>

namespace cs::gui::detail {

/// Implements browser requests to system resources etc.
class RequestHandler : public CefRequestHandler {

 public:
  /// Implements to ignore certificate errors.
  bool OnCertificateError(CefRefPtr<CefBrowser> browser, cef_errorcode_t cert_error,
      CefString const& request_url, CefRefPtr<CefSSLInfo> ssl_info,
      CefRefPtr<CefRequestCallback> callback) override;

  CefRefPtr<CefResourceRequestHandler> GetResourceRequestHandler(CefRefPtr<CefBrowser> /*browser*/,
      CefRefPtr<CefFrame> /*frame*/, CefRefPtr<CefRequest> /*request*/, bool /*is_navigation*/,
      bool /*is_download*/, const CefString& /*request_initiator*/,
      bool& /*disable_default_handling*/) override {
    return mResourceRequestHandler;
  };

 private:
  IMPLEMENT_REFCOUNTING(RequestHandler);
  CefRefPtr<CefResourceRequestHandler> mResourceRequestHandler = new ResourceRequestHandler();
};

} // namespace cs::gui::detail

#endif // CS_GUI_DETAIL_REQUESTHANDLER_HPP
