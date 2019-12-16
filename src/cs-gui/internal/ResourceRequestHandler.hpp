////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

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
