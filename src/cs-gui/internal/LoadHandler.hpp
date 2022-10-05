////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

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
