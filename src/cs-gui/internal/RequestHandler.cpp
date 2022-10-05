////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "RequestHandler.hpp"

#include "../logger.hpp"

namespace cs::gui::detail {

////////////////////////////////////////////////////////////////////////////////////////////////////

bool RequestHandler::OnCertificateError(CefRefPtr<CefBrowser> /*browser*/,
    cef_errorcode_t /*cert_error*/, CefString const& /*request_url*/,
    CefRefPtr<CefSSLInfo> /*ssl_info*/, CefRefPtr<CefRequestCallback> callback) {

  logger().warn("Detected a certificate error in Chromium Embedded Framework. Continuing...");

  callback->Continue(true);
  return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cs::gui::detail
