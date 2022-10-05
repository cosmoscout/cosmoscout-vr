////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CS_GUI_DETAIL_RESOURCEREQUESTHANDLER_HPP
#define CS_GUI_DETAIL_RESOURCEREQUESTHANDLER_HPP

#include <include/cef_client.h>

namespace cs::gui::detail {

/// Implements browser requests to system resources etc.
class ResourceRequestHandler : public CefResourceRequestHandler {

 public:
  /// Forwards file loading to the file system.
  CefRefPtr<CefResourceHandler> GetResourceHandler(CefRefPtr<CefBrowser> browser,
      CefRefPtr<CefFrame> frame, CefRefPtr<CefRequest> request) override;

 private:
  IMPLEMENT_REFCOUNTING(ResourceRequestHandler);
};

} // namespace cs::gui::detail

#endif // CS_GUI_DETAIL_RESOURCEREQUESTHANDLER_HPP
