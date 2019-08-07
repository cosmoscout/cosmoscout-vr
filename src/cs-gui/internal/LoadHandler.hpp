////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CS_GUI_DETAIL_LOADHANDLER_HPP
#define CS_GUI_DETAIL_LOADHANDLER_HPP

#include <include/cef_client.h>

namespace cs::gui::detail {

/// Handles browser loading events.
class LoadHandler : public CefLoadHandler {

 public:
  /// Lets pauses the execution until the browser has loaded its contents.
  void WaitForFinishedLoading() const;

  /// Gets called when the loading state changes. And halts the blocking of execution in
  /// WaitForFinishedLoading().
  void OnLoadingStateChange(
      CefRefPtr<CefBrowser> browser, bool isLoading, bool canGoBack, bool canGoForward) override;

 private:
  IMPLEMENT_REFCOUNTING(LoadHandler);

  bool mInitialized = false;
};

} // namespace cs::gui::detail

#endif // CS_GUI_DETAIL_LOADHANDLER_HPP
