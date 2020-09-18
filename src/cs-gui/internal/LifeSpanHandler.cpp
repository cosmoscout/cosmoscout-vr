////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "LifeSpanHandler.hpp"

#include "../../cs-utils/utils.hpp"
#include "../logger.hpp"

namespace cs::gui::detail {

////////////////////////////////////////////////////////////////////////////////////////////////////

bool LifeSpanHandler::OnBeforePopup(CefRefPtr<CefBrowser> /*browser*/,
    CefRefPtr<CefFrame> /*frame*/, const CefString& target_url,
    const CefString& /*target_frame_name*/, WindowOpenDisposition /*target_disposition*/,
    bool /*user_gesture*/, const CefPopupFeatures& /*popupFeatures*/, CefWindowInfo& /*windowInfo*/,
    CefRefPtr<CefClient>& /*client*/, CefBrowserSettings& /*settings*/,
    CefRefPtr<CefDictionaryValue>& /*extra_info*/, bool* /*no_javascript_access*/) {

  auto url = target_url.ToString();

  logger().info("Opening the following URL in an external browser: {}.", url);

  int         result = 0;
  std::string command;

  // Open links in the default OS browser to avoid breaking the CosmoScout GUI.
  if constexpr (utils::HostOS == utils::OS::eLinux) {
    command = "xdg-open " + url + " >/dev/null 2>&1";
    result  = system(command.c_str());
  } else if constexpr (utils::HostOS == utils::OS::eWindows) {
    command = "start " + url + " > nul";
    result  = system(command.c_str());
  }

  if (result != 0) {
    logger().warn("Failed to execute system command '{}'!", command);
  }

  return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void LifeSpanHandler::OnAfterCreated(CefRefPtr<CefBrowser> browser) {
  mBrowser = browser;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cs::gui::detail
