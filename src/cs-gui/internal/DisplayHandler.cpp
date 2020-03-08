////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "DisplayHandler.hpp"

#include <iostream>
#include <spdlog/spdlog.h>

namespace cs::gui::detail {

////////////////////////////////////////////////////////////////////////////////////////////////////

bool DisplayHandler::OnConsoleMessage(CefRefPtr<CefBrowser> browser, cef_log_severity_t level,
    CefString const& message, CefString const& source, int line) {

  if (level == LOGSEVERITY_VERBOSE) {
    spdlog::trace(message.ToString());
  } else if (level == LOGSEVERITY_DEBUG) {
    spdlog::debug(message.ToString());
  } else if (level == LOGSEVERITY_WARNING) {
    spdlog::warn(message.ToString());
  } else if (level == LOGSEVERITY_ERROR) {
    spdlog::error(message.ToString());
  } else if (level == LOGSEVERITY_FATAL) {
    spdlog::critical(message.ToString());
  } else {
    spdlog::info(message.ToString());
  }

  return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cs::gui::detail
