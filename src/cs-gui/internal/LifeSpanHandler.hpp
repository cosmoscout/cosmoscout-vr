////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CS_GUI_DETAIL_LIFESPANHANDLER_HPP
#define CS_GUI_DETAIL_LIFESPANHANDLER_HPP

#include <include/cef_client.h>

namespace cs::gui::detail {

/// Implements actions related to the browsers lifetime events.
class LifeSpanHandler : public CefLifeSpanHandler {

 public:
  CefRefPtr<CefBrowser> const& GetBrowser() const {
    return mBrowser;
  }

  /// Gets called when a popup or new window is to be created. We always open them in the system's
  /// default browser.
  bool OnBeforePopup(CefRefPtr<CefBrowser> browser, CefRefPtr<CefFrame> frame,
      const CefString& target_url, const CefString& target_frame_name,
      WindowOpenDisposition target_disposition, bool user_gesture,
      const CefPopupFeatures& popupFeatures, CefWindowInfo& windowInfo,
      CefRefPtr<CefClient>& client, CefBrowserSettings& settings,
      CefRefPtr<CefDictionaryValue>& extra_info, bool* no_javascript_access) override;

  /// Gets called after the browser is created and sets the browser field.
  void OnAfterCreated(CefRefPtr<CefBrowser> browser) override;

  /// Gets called, when the browser is destroyed.
  bool DoClose(CefRefPtr<CefBrowser> /*browser*/) override {
    mBrowser = nullptr;
    return false;
  }

 private:
  IMPLEMENT_REFCOUNTING(LifeSpanHandler);

  CefRefPtr<CefBrowser> mBrowser;
};

} // namespace cs::gui::detail

#endif // CS_GUI_DETAIL_LIFESPANHANDLER_HPP
