////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CS_GUI_DETAIL_REQUESTHANDLER_HPP
#define CS_GUI_DETAIL_REQUESTHANDLER_HPP

#include <include/cef_client.h>

namespace cs::gui::detail {

/// Implements browser requests to system resources etc.
class RequestHandler : public CefRequestHandler {

 public:
  /// Forwards file loading to the file system.
  CefRefPtr<CefResourceHandler> GetResourceHandler(CefRefPtr<CefBrowser> browser,
      CefRefPtr<CefFrame> frame, CefRefPtr<CefRequest> request) override;

  /// Implements to ignore certificate errors.
  bool OnCertificateError(CefRefPtr<CefBrowser> browser, cef_errorcode_t cert_error,
      CefString const& request_url, CefRefPtr<CefSSLInfo> ssl_info,
      CefRefPtr<CefRequestCallback> callback) override;

 private:
  IMPLEMENT_REFCOUNTING(RequestHandler);
};

} // namespace cs::gui::detail

#endif // CS_GUI_DETAIL_REQUESTHANDLER_HPP
